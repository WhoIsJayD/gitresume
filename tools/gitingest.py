"""
Repository ingestion and analysis tool.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Set

import aiofiles

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
IGNORE_DIRS: Set[str] = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "build", "dist",
    ".pytest_cache", ".mypy_cache", "target", "out", "docs"
}
IGNORE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".db", ".sqlite3", ".log", ".exe", ".bin", ".so",
    ".dll", ".o", ".a", ".obj", ".lib", ".zip", ".tar", ".gz", ".rar", ".md",
    ".txt"
}
MAX_FILES_TO_PROCESS: int = 5000
MAX_FILE_SIZE_BYTES: int = 1 * 1024 * 1024
TEXT_FILE_EXTENSIONS: Set[str] = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.gitignore', '.dockerignore', '.editorconfig'
}

# --- Tree-sitter Initialization ---
PARSER: Optional[Any] = None
LANGUAGE_MAP: Dict[str, Any] = {}
try:
    from tree_sitter import Language, Parser

    PARSER = Parser()

    language_loaders = {
        '.py': lambda: Language(__import__('tree_sitter_python').language(), 'python'),
        '.js': lambda: Language(__import__('tree_sitter_javascript').language(), 'javascript'),
        '.jsx': lambda: Language(__import__('tree_sitter_javascript').language(), 'javascript'),
        '.go': lambda: Language(__import__('tree_sitter_go').language(), 'go'),
        '.rs': lambda: Language(__import__('tree_sitter_rust').language(), 'rust'),
        '.java': lambda: Language(__import__('tree_sitter_java').language(), 'java'),
        '.c': lambda: Language(__import__('tree_sitter_c').language(), 'c'),
        '.cpp': lambda: Language(__import__('tree_sitter_cpp').language(), 'cpp'),
        '.h': lambda: Language(__import__('tree_sitter_c').language(), 'c'),
        '.hpp': lambda: Language(__import__('tree_sitter_cpp').language(), 'cpp'),
        '.cs': lambda: Language(__import__('tree_sitter_c_sharp').language(), 'c_sharp'),
    }

    try:
        from tree_sitter_typescript import language_typescript, language_tsx

        language_loaders['.ts'] = lambda: Language(language_typescript(), 'typescript')
        language_loaders['.tsx'] = lambda: Language(language_tsx(), 'tsx')
    except (ImportError, AttributeError):
        logger.warning("tree-sitter-typescript not found or invalid. TypeScript analysis disabled.")

    for ext, loader in language_loaders.items():
        try:
            LANGUAGE_MAP[ext] = loader()
        except Exception as e:
            logger.warning(f"Could not load tree-sitter language for '{ext}'. Error: {e}")

    if not LANGUAGE_MAP:
        PARSER = None
    else:
        logger.info(f"Tree-sitter initialized for: {list(LANGUAGE_MAP.keys())}")

except ImportError:
    logger.warning("Tree-sitter library not found. Falling back to text-only analysis.")
except Exception as e:
    logger.error(f"An unexpected error during tree-sitter initialization: {e}")


def _extract_ast_metrics(tree: Any, lang: Any) -> Dict[str, Any]:
    """Extracts metrics like function and class counts using language-specific queries."""
    if not tree or not lang: return {}

    # Language-specific queries for code analysis
    queries = {
        'python': {"functions": "(function_definition) @f", "classes": "(class_definition) @c"},
        'javascript': {"functions": "[(function_declaration) (arrow_function) (method_definition)] @f",
                       "classes": "(class_declaration) @c"},
        'go': {"functions": "(function_declaration) @f", "classes": "(type_spec name: (type_identifier)) @c"},
        'rust': {"functions": "(function_item) @f", "classes": "[(struct_item) (enum_item)] @c"},
        'java': {"functions": "(method_declaration) @f", "classes": "(class_declaration) @c"},
        'c': {"functions": "(function_definition) @f", "classes": "[(struct_specifier) (enum_specifier)] @c"},
        'cpp': {"functions": "(function_definition) @f", "classes": "[(class_specifier) (struct_specifier)] @c"},
        'c_sharp': {"functions": "(method_declaration) @f", "classes": "[(class_declaration) (struct_declaration)] @c"},
    }
    queries['typescript'] = queries['javascript']
    queries['tsx'] = queries['javascript']

    lang_name = lang.name
    lang_queries = queries.get(lang_name, {})

    def count_captures(query_str):
        if not query_str: return 0
        try:
            query = lang.query(query_str)
            return len(query.captures(tree.root_node))
        except Exception:
            return 0

    return {'functions': count_captures(lang_queries.get('functions')),
            'classes': count_captures(lang_queries.get('classes'))}


async def _process_file(file_path: Path, repo_root: Path) -> Optional[Dict[str, Any]]:
    try:
        def get_size():
            return file_path.stat().st_size

        file_size = await asyncio.to_thread(get_size)

        if file_size > MAX_FILE_SIZE_BYTES or (
                file_size == 0 and file_path.suffix.lower() not in TEXT_FILE_EXTENSIONS):
            return None

        rel_path_str = file_path.relative_to(repo_root).as_posix()
        file_info = {'path': rel_path_str, 'size': file_size, 'extension': file_path.suffix.lower(),
                     'content_preview': "", 'metrics': {}}

        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        file_info['content_preview'] = content[:2000]

        if PARSER and file_info['extension'] in LANGUAGE_MAP:
            parser_lang = LANGUAGE_MAP[file_info['extension']]

            def parse_sync():
                PARSER.set_language(parser_lang)
                return PARSER.parse(bytes(content, "utf8"))

            tree = await asyncio.to_thread(parse_sync)
            if tree and tree.root_node and not tree.root_node.has_error:
                file_info['metrics'] = _extract_ast_metrics(tree, parser_lang)

        return file_info
    except Exception as e:
        logger.warning(f"Error processing file '{file_path}': {e}")
        return None


async def gitingest_tool(repo_path: str) -> Dict[str, Any]:
    repo_root = Path(repo_path).resolve()
    if not (repo_root.exists() and repo_root.is_dir() and (repo_root / ".git").exists()):
        return {"success": False, "error": f"Path '{repo_path}' is not a valid Git repository."}

    logger.info(f"Starting ingestion of repository: {repo_path}")
    tasks, file_paths_to_process, tree_structure = [], [], []

    file_count = 0
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        root_path = Path(root)
        rel_root = root_path.relative_to(repo_root).as_posix()
        if rel_root == ".":
            rel_root = ""
        tree_structure.append(f"{rel_root}/")

        for file_name in files:
            if file_count >= MAX_FILES_TO_PROCESS:
                break
            file_path = root_path / file_name
            if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                continue
            file_paths_to_process.append(file_path)
            file_count += 1
        if file_count >= MAX_FILES_TO_PROCESS:
            break

    for file_path in file_paths_to_process:
        tasks.append(_process_file(file_path, repo_root))
    processed_files_info = await asyncio.gather(*tasks)

    summary = {"total_files_processed": 0, "total_size_bytes": 0, "file_types": {},
               "code_metrics": {"total_functions": 0, "total_classes": 0, "languages": {}}}
    all_content = {}
    for file_info in filter(None, processed_files_info):
        summary["total_files_processed"] += 1
        summary["total_size_bytes"] += file_info['size']
        ext = file_info['extension'] or ".other"
        summary["file_types"][ext] = summary["file_types"].get(ext, 0) + 1
        all_content[file_info['path']] = file_info['content_preview']
        if file_info['metrics']:
            metrics, lang_metrics = file_info['metrics'], summary["code_metrics"]["languages"].setdefault(ext,
                                                                                                          {'files': 0,
                                                                                                           'functions': 0,
                                                                                                           'classes': 0})
            summary["code_metrics"]["total_functions"] += metrics.get('functions', 0)
            summary["code_metrics"]["total_classes"] += metrics.get('classes', 0)
            lang_metrics['files'] += 1
            lang_metrics['functions'] += metrics.get('functions', 0)
            lang_metrics['classes'] += metrics.get('classes', 0)

    if summary["total_files_processed"] == 0:
        return {"success": False, "error": "No processable files were found."}

    logger.info(f"Ingestion complete for '{repo_path}'. Processed {summary['total_files_processed']} files.")
    return {"success": True, "tree": "\n".join(sorted(tree_structure)), "summary": summary, "content": all_content}

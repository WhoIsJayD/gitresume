# GitResume 🚀

*Transform your GitHub repositories into professional, ATS-optimized resumes using AI.*

![GitResume Screenshot](static/screenshot.png)  
[![Live: gitresume.live](https://img.shields.io/badge/Live-gitresume.live-brightgreen?logo=google-chrome)](https://gitresume.live)  
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)  
[![Docker](https://img.shields.io/badge/Docker-20.10-blue.svg)](https://www.docker.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)  
[![GitHub Issues](https://img.shields.io/github/issues/whoisjayd/gitresume)](https://github.com/whoisjayd/gitresume/issues)  
[![GitHub Stars](https://img.shields.io/github/stars/whoisjayd/gitresume)](https://github.com/whoisjayd/gitresume/stargazers)  
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/whoisjayd/gitresume)

---

## Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [What You Get](#what-you-get)
- [Who Is This For?](#who-is-this-for)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Configuration](#configuration)
- [Resume Output](#resume-output)
- [FAQ](#frequently-asked-questions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Architecture Overview](#architecture-overview)
- [API Endpoints](#api-endpoints)
- [Module Breakdown](#module-breakdown)
- [Why GitResume?](#why-gitresume)
- [Contact](#contact)

---

## 🌐 Live Demo

Experience GitResume in action: [https://gitresume.live](https://gitresume.live)

---

## 📝 Project Overview

GitResume is a production-ready FastAPI web application that leverages AI to convert your GitHub repositories into polished, ATS-friendly resume content. By analyzing your codebase with advanced parsing techniques, it extracts technical achievements and transforms them into impactful resume sections. Tailored for engineers at all career stages, GitResume supports multiple AI providers and is fully containerized for seamless deployment.

---

## 🎯 What You Get

Turn this:

> A repo with FastAPI backend, Redis caching, and Docker deployment.

Into this:

- Developed scalable backend using FastAPI and Redis for caching.  
- Implemented CI/CD with Docker and GitHub Actions.  
- Designed microservices structure for modular development.

---

## 💼 Who Is This For?

- 🧑‍🎓 *Students*: Turn class projects into resume-ready experience.
- 👩‍💻 *Developers*: Showcase contributions with tailored resume bullets.
- 🧑‍💼 *Job Seekers*: Get ATS-optimized content in minutes.
- 👥 *Recruiters*: Generate summaries from candidate GitHub links.

---

## ✨ Key Features

- *AI-Powered Resume Creation:* Converts GitHub repositories into ATS-friendly resumes.
- *Smart Code Parsing:* Uses tree-sitter to extract tech stack, structure, and key contributions.
- *Customizable & Tailored Output:* Adapts resumes to specific job descriptions and formats.
- *Secure & Scalable:* GitHub OAuth, rate limiting, Redis support, and production-ready Docker setup.

---

## 🚀 Getting Started

### Prerequisites

- *Python 3.11* (for local development)
- *Docker* (recommended for production)
- *Git* (for cloning repositories)
- *Redis* (optional, for rate limiting and session management)
- *API Keys*:
    - GitHub OAuth (GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GITHUB_TOKEN)
    - AI Providers (Gemini, OpenAI, Groq, Claude)

### Installation (Local)

1. *Clone the Repository:*
   bash
   git clone https://github.com/whoisjayd/gitresume.git
   cd gitresume
   
2. *Set Up Virtual Environment:*
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
3. *Install Dependencies:*
   bash
   pip install --upgrade pip
   pip install -r requirements.txt
   
4. *Configure Environment Variables:*
    - Copy .env.example to .env and populate with your API keys and secrets.
    - Alternatively, use env.yaml for configuration.
5. *Run the Application:*
   bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   
   Access the app at [http://localhost:8080](http://localhost:8080).

### Docker Deployment

1. *Build the Docker Image:*
   bash
   docker build -t gitresume .
   
2. *Run the Container:*
   bash
   docker run --env-file .env -p 8080:8080 gitresume
   
   Alternatively, mount env.yaml for configuration:
   bash
   docker run -v $(pwd)/env.yaml:/app/env.yaml -p 8080:8080 gitresume
   

---

## 🛠 Tech Stack

| Component            | Technology                                         |
|----------------------|----------------------------------------------------|
| *Backend*          | Python 3.11, FastAPI, Starlette, Pydantic, SlowAPI |
| *Frontend*         | Jinja2 Templates, Tailwind CSS (CDN)               |
| *AI Providers*     | Gemini, OpenAI, Groq, Claude                       |
| *Caching*          | Redis (optional)                                   |
| *Containerization* | Docker, Uvicorn                                    |
| *Code Parsing*     | Tree-sitter                                        |

---

## 🔧 Configuration

- *Environment Variables:*
    - Copy .env.example to .env or create env.yaml.
    - Key variables include:
        - GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GITHUB_TOKEN
        - REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD
        - AI_PROVIDER (options: gemini, openai, groq, claude)
        - API keys for AI providers
        - SESSION_SECRET_KEY, CALLBACK_URL, ENVIRONMENT
    - Both .env and env.yaml are supported for local and Docker setups.
- **Example .env:**
  bash
  GITHUB_CLIENT_ID=your_client_id
  GITHUB_CLIENT_SECRET=your_client_secret
  AI_PROVIDER=gemini
  GEMINI_API_KEY=your_gemini_key
  ENVIRONMENT=production
  

---

## 📄 Resume Output

GitResume generates structured JSON output optimized for ATS systems and easy integration into resume templates. The schema includes:

- project_title: Project name (string)
- tech_stack: List of technologies used (array)
- bullet_points: 4-6 concise achievement statements (array)
- additional_notes: Unique technical setup, deployment strategies, or noteworthy engineering decisions (string)
- future_plans: Logical next features or enhancements (string)
- potential_advancements: Advanced architectural improvements or optimizations (string)
- interview_questions: Array of objects with question, answer, and category

*Example Output:*

json
{
  "project_title": "E-Commerce Platform",
  "tech_stack": ["Python", "Redis"],
  "bullet_points": [
    "Developed scalable backend using FastAPI and Redis for caching.",
    "Integrated secure payment gateway and OAuth authentication.",
    "Implemented CI/CD pipeline with Docker and GitHub Actions.",
    "Optimized database queries, reducing latency by 30%."
  ],
  "additional_notes": "Designed modular microservices architecture for rapid feature deployment.",
  "future_plans": "Add automated testing and expand API documentation.",
  "potential_advancements": "Utilize event-driven design with message queues for order processing.",
  "interview_questions": [
    {
      "question": "How did you ensure scalability in the backend?",
      "answer": "By leveraging FastAPI's async capabilities and Redis for caching.",
      "category": "Backend Architecture"
    }
  ]
}


---

## ❓ Frequently Asked Questions

- *Is my GitHub data stored?*  
  No, repositories are cloned temporarily and deleted after analysis.
- *Which AI models are supported?*  
  Gemini, OpenAI, Groq, and Claude are fully compatible which can be configured via `.env`.
- *Can I analyze private repositories?*  
  Yes, after authenticating via GitHub OAuth.
- *How do I deploy in production?*  
  Use the provided Docker setup with environment variables configured.
- *How do I report bugs or suggest features?*  
  Open an issue on [GitHub Issues](https://github.com/whoisjayd/gitresume/issues).
- *How do I contribute?*  
  See the [Contributing](#contributing) section below for guidelines and steps.

---

## 🤝 Contributing

We welcome contributions! Follow these steps:

1. Fork and clone the repository.
2. Create a feature branch: git checkout -b feature/your-feature.
3. Implement and test your changes locally.
4. Push to your fork and submit a pull request.

*Contribution Guidelines:*

- Follow PEP 8 for Python code.
- Include tests for new features (use pytest).
- Update documentation as needed.
- Be respectful and inclusive in all interactions (see [Code of Conduct](CODE_OF_CONDUCT.md)).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- *Core Technologies:* [Python](https://www.python.org/), [FastAPI](https://fastapi.tiangolo.com/), [Tree-sitter](https://tree-sitter.github.io/), [Jinja2](https://jinja.palletsprojects.com/), [Tailwind CSS](https://tailwindcss.com/), [Redis](https://redis.io/), [Uvicorn](https://www.uvicorn.org/), [Docker](https://www.docker.com/)
- *AI Providers:* [Google Gemini](https://cloud.google.com/generative-ai), [OpenAI](https://openai.com/), [Groq](https://groq.com/), [Anthropic Claude](https://www.anthropic.com/)

---

## 🏗 Architecture Overview

GitResume follows a modular, scalable architecture:

- *Frontend:* Jinja2 templates styled with Tailwind CSS, served by FastAPI.
- *Backend:* FastAPI handles routing, API endpoints, and WebSocket connections.
- *Authentication:* GitHub OAuth for secure access to public and private repos.
- *Code Analysis:* Tree-sitter parses codebases for structure and technology insights.
- *AI Integration:* Modular support for Gemini, OpenAI, Groq, and Claude.
- *Caching & Rate Limiting:* Redis (optional) for session management and performance.
- *Deployment:* Docker with multi-stage builds for minimal, secure images.

---

## 🔗 API Endpoints

| Endpoint         | Method   | Description                     |
|------------------|----------|---------------------------------|
| /              | GET      | Home page                       |
| /              | POST     | Generate resume from repo URL   |
| /login         | GET      | Initiate GitHub OAuth           |
| /callback      | GET      | Handle OAuth callback           |
| /logout        | GET      | Log out                         |
| /health        | GET      | Health check endpoint           |
| /{user}/{repo} | GET/POST | Dynamic repo analysis           |
| /ws/           | WS       | WebSocket for real-time updates |

---

## 📦 Module Breakdown

| Module                    | Purpose                                      |
|---------------------------|----------------------------------------------|
| app.py                  | Core FastAPI app with routing and middleware |
| tools/create_resume.py  | Orchestrates AI resume generation            |
| tools/git_operations.py | Manages repo cloning and validation          |
| tools/gitingest.py      | Parses and summarizes codebases              |
| tools/grammar_check.py  | Ensures high-quality AI text output          |
| tools/api_utils.py      | Integrates with AI provider APIs             |
| tools/utils.py          | General utility functions                    |

---

## 🌟 Why GitResume?

- *No More Blank Pages:* Start your resume with content generated from real code.
- *Designed for Engineers:* Resume bullets highlight your actual skills and impact.
- *Recruiter-Ready:* Outputs are optimized for ATS and easy formatting.
- *Plug-and-Play:* Works with public and private repos, compatible with multiple AI, and Docker.
- *Open Source:* Built for the community, contributions welcome!

---

## 📬 Contact

Created with ❤ by [Jaydeep Solanki](https://github.com/whoisjayd).  
Have questions or feedback? Reach out via [GitHub Issues](https://github.com/whoisjayd/gitresume/issues).
# Autonomous Research Paper Analyzer

Extract insights with intelligent research analysis. This project leverages autonomous AI agents powered by [CrewAI](https://crewai.com) to search, analyze, and summarize academic research papers. Get comprehensive research briefs with a React + Vite frontend for intuitive paper discovery and analysis.

## Features

- **Autonomous Paper Discovery**: AI agents automatically search and retrieve academic papers from multiple sources
- **Intelligent Ranking**: Papers are ranked by relevance to your research topic
- **Comprehensive Analysis**: Extract key insights, methodologies, and findings from papers
- **Beautiful UI**: Modern React + Vite frontend for easy interaction
- **FastAPI Backend**: RESTful API with async support for scalable operations
- **Configurable Analysis**: Control search breadth, analysis depth, and report length

## Tech Stack

- **Backend**: Python, CrewAI, FastAPI, Uvicorn
- **Frontend**: React, Vite, ESLint
- **AI**: OpenAI LLMs via CrewAI
- **Tools**: Tavily Search API, PyPDF
- **Package Manager**: UV

## Prerequisites

- Python >=3.10 <3.14
- Node.js 16+ (for frontend)
- UV (Python package manager)
- OpenAI API Key
- Tavily API Key (for paper search)

## Installation

### Backend Setup

1. **Install UV** (if not already installed):
```bash
pip install uv
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Set up environment variables**:
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
```

## Configuration

### AI Agents & Tasks

- **Agents**: Defined in `src/autonomous_research_paper_analyzer_v2/config/agents.yaml`
- **Tasks**: Defined in `src/autonomous_research_paper_analyzer_v2/config/tasks.yaml`

Modify these YAML files to customize agent behaviors and task workflows.

## Running the Project

### Option 1: Backend Only

```bash
uvicorn autonomous_research_paper_analyzer_v2.api:app --reload --port 8000
```

### Option 2: Backend + Frontend (Development)

**Terminal 1 - Backend**:
```bash
uvicorn autonomous_research_paper_analyzer_v2.api:app --reload --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

Access the app at `http://localhost:5173`

### Option 3: Using CrewAI CLI

```bash
crewai run
```

This generates a `report.md` with analysis results.

## API Reference

### Endpoints

#### Health Check
- **GET** `/api/health`

#### Synchronous Analysis
- **POST** `/api/analyze`
- **Body**:
```json
{
  "topic": "machine learning",
  "max_papers": 60,
  "analysis_papers": 20,
  "report_words": 800
}
```

#### Asynchronous Analysis
- **POST** `/api/analyze/async`
- Returns: `job_id`

#### Job Status
- **GET** `/api/jobs/{job_id}`

#### Cancel Job
- **POST** `/api/jobs/{job_id}/cancel`

### Parameters

- `topic`: Research topic to analyze
- `max_papers`: Total papers to retrieve (up to 200)
- `analysis_papers`: Papers to analyze in depth (up to 60)
- `report_words`: Final report length (500-1200)

## Project Structure

```
├── src/autonomous_research_paper_analyzer_v2/
│   ├── api.py              # FastAPI application
│   ├── crew.py             # CrewAI setup
│   ├── main.py             # CLI entry point
│   ├── config/
│   │   ├── agents.yaml     # Agent definitions
│   │   └── tasks.yaml      # Task definitions
│   └── tools/
│       └── custom_tool.py  # Custom tools
├── frontend/               # React + Vite app
├── tests/                  # Test suite
└── pyproject.toml         # Python dependencies
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key

# Optional
VITE_API_BASE_URL=http://localhost:8000
```

## Development

### Running Tests

```bash
pytest
```

### Linting & Formatting

Frontend:
```bash
cd frontend
npm run lint
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Check the [CrewAI Documentation](https://docs.crewai.com)
- Visit [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
- Join [CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)


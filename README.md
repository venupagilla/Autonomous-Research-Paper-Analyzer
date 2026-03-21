# AutonomousResearchPaperAnalyzerV2 Crew

Welcome to the AutonomousResearchPaperAnalyzerV2 Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/autonomous_research_paper_analyzer_v2/config/agents.yaml` to define your agents
- Modify `src/autonomous_research_paper_analyzer_v2/config/tasks.yaml` to define your tasks
- Modify `src/autonomous_research_paper_analyzer_v2/crew.py` to add your own logic, tools and specific args
- Modify `src/autonomous_research_paper_analyzer_v2/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the Autonomous_Research_Paper_Analyzer_v2 Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Frontend + Backend Integration

### Start the Backend API

Run from the project root:

```bash
uvicorn autonomous_research_paper_analyzer_v2.api:app --reload --port 8000
```

API endpoints:

- `GET /api/health`
- `POST /api/analyze`
- `POST /api/analyze/async`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`

Example request body:

```json
{
	"topic": "autonomous research agents",
	"max_papers": 60,
	"analysis_papers": 20,
	"report_words": 800
}
```

Token-efficient scaling pattern:

- `max_papers`: retrieval breadth (up to 200)
- `analysis_papers`: focused subset for deep analysis (up to 60, auto-capped at `max_papers`)
- `report_words`: controls final synthesis size (500-1200)

For larger coverage with controlled cost, use high `max_papers` (e.g., 80-150) and moderate
`analysis_papers` (e.g., 15-30).

Async flow:

1. Call `POST /api/analyze/async` with the same request body.
2. Poll `GET /api/jobs/{job_id}` until `status` is `completed`, `failed`, or `canceled`.
3. Read final payload from `result` when status is `completed`.

Frontend behavior:

- Renders `report_markdown` using a markdown viewer.
- Uses request timeout for long sync calls.
- Supports canceling running requests and async jobs from the UI.

### Start the React Frontend

Run from `frontend`:

```bash
npm run dev
```

Optional frontend env var:

- `VITE_API_BASE_URL` (default: `http://localhost:8000`)

## Understanding Your Crew

The Autonomous_Research_Paper_Analyzer_v2 Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the AutonomousResearchPaperAnalyzerV2 Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.

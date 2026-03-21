import os
import time

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import ArxivPaperTool, TavilySearchTool
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AutonomousResearchPaperAnalyzerV2():
    """AutonomousResearchPaperAnalyzerV2 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _pause_between_tasks(self, _task_output) -> None:
        """Pause after each task to avoid bursting token-per-minute limits."""
        self._completed_tasks = getattr(self, "_completed_tasks", 0) + 1
        if self._completed_tasks < len(self.tasks):
            delay_seconds = int(os.getenv("AGENT_DELAY_SECONDS", "60"))
            print(f"\nRate-limit protection: waiting {delay_seconds} seconds before the next agent...\n")
            time.sleep(delay_seconds)

    def _small_llm(self, temperature: float = 0.1) -> LLM:
        """Low-cost model for extraction and ranking tasks."""
        return LLM(
            model=os.getenv("SMALL_LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    def _strong_llm(self, temperature: float = 0.2) -> LLM:
        """Stronger model for synthesis and final report generation."""
        return LLM(
            model=os.getenv("STRONG_LLM_MODEL", "gpt-4.1"),
            temperature=temperature,
        )

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def paper_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['paper_researcher'], # type: ignore[index]
            # Tavily is the only web-search provider used in this project.
            tools=[ArxivPaperTool(), TavilySearchTool()],
            llm=self._small_llm(temperature=0.0),
            verbose=False,
        )

    @agent
    def paper_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['paper_analyst'], # type: ignore[index]
            llm=self._small_llm(temperature=0.1),
            verbose=False,
        )
    
    @agent
    def knowledge_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['knowledge_synthesizer'], # type: ignore[index]
            llm=self._strong_llm(temperature=0.2),
            verbose=False,
        )
    
    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'], # type: ignore[index]
            llm=self._strong_llm(temperature=0.2),
            verbose=False,
        )
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def fetch_papers_task(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_papers_task'], # type: ignore[index]
        )

    @task
    def analyze_papers_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_papers_task'], # type: ignore[index]
        )

    @task
    def synthesize_knowledge_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_knowledge_task'], # type: ignore[index]
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AutonomousResearchPaperAnalyzerV2 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        self._completed_tasks = 0
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            task_callback=self._pause_between_tasks,
            verbose=False,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

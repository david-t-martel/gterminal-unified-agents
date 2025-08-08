#!/usr/bin/env python3
"""MCP Server for Master Architect Agent.

Provides MCP server interface for the Master Architect Agent service.
"""

import logging
from typing import Any

from gterminal.agents.master_architect_agent import MasterArchitectService
from mcp.server import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("master-architect")

# Initialize the agent service
agent_service = MasterArchitectService()


@mcp.tool()
async def design_system(
    requirements: str,
    project_type: str = "web-application",
    scalability: str = "medium",
    include_diagrams: bool = True,
) -> dict[str, Any]:
    """Design a complete system architecture based on requirements.

    Args:
        requirements: System requirements and constraints
        project_type: Type of project ('web-application', 'mobile-app', 'microservices', 'data-pipeline')
        scalability: Scalability requirements ('low', 'medium', 'high', 'extreme')
        include_diagrams: Whether to generate architecture diagrams

    Returns:
        Complete system architecture design

    """
    try:
        request = {
            "requirements": requirements,
            "project_type": project_type,
            "scalability": scalability,
            "include_diagrams": include_diagrams,
        }

        return await agent_service.design_system(request)
    except Exception as e:
        logger.exception(f"System design failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def recommend_tech_stack(
    requirements: str,
    project_type: str,
    team_expertise: str = "",
    budget_constraints: str = "medium",
) -> dict[str, Any]:
    """Recommend optimal technology stack for a project.

    Args:
        requirements: Project requirements and goals
        project_type: Type of project
        team_expertise: Comma-separated technologies the team knows
        budget_constraints: Budget level ('low', 'medium', 'high', 'enterprise')

    Returns:
        Technology stack recommendations with justifications

    """
    try:
        # Parse team_expertise from comma-separated string
        team_expertise_list = (
            [item.strip() for item in team_expertise.split(",") if item.strip()]
            if team_expertise
            else []
        )

        # Parse team_expertise from comma-separated string
        team_expertise_list = (
            [item.strip() for item in team_expertise.split(",") if item.strip()]
            if team_expertise
            else []
        )

        # Parse team_expertise from comma-separated string
        team_expertise_list = (
            [item.strip() for item in team_expertise.split(",") if item.strip()]
            if team_expertise
            else []
        )

        # Parse team_expertise from comma-separated string
        team_expertise_list = (
            [item.strip() for item in team_expertise.split(",") if item.strip()]
            if team_expertise
            else []
        )

        # Parse team expertise from comma-separated string
        team_expertise_list = (
            [tech.strip() for tech in team_expertise.split(",") if tech.strip()]
            if team_expertise
            else []
        )

        request = {
            "requirements": requirements,
            "project_type": project_type,
            "team_expertise": team_expertise_list,
            "budget_constraints": budget_constraints,
        }

        return await agent_service.recommend_technology_stack(request)
    except Exception as e:
        logger.exception(f"Tech stack recommendation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def review_architecture(
    project_path: str,
    focus_areas: str = "",
    suggest_improvements: bool = True,
) -> dict[str, Any]:
    """Review existing system architecture and provide recommendations.

    Args:
        project_path: Path to the project to review
        focus_areas: Comma-separated areas to focus on (e.g., 'scalability,security,performance')
        suggest_improvements: Whether to suggest improvements

    Returns:
        Architecture review results and recommendations

    """
    try:
        # Parse focus_areas from comma-separated string
        focus_areas_list = (
            [item.strip() for item in focus_areas.split(",") if item.strip()] if focus_areas else []
        )

        # Parse focus_areas from comma-separated string
        focus_areas_list = (
            [item.strip() for item in focus_areas.split(",") if item.strip()] if focus_areas else []
        )

        # Parse focus_areas from comma-separated string
        focus_areas_list = (
            [item.strip() for item in focus_areas.split(",") if item.strip()] if focus_areas else []
        )

        # Parse focus_areas from comma-separated string
        focus_areas_list = (
            [item.strip() for item in focus_areas.split(",") if item.strip()] if focus_areas else []
        )

        # Parse focus areas from comma-separated string
        focus_areas_list = (
            [area.strip() for area in focus_areas.split(",") if area.strip()]
            if focus_areas
            else ["scalability", "security", "maintainability"]
        )

        request = {
            "project_path": project_path,
            "focus_areas": focus_areas_list,
            "suggest_improvements": suggest_improvements,
        }

        return await agent_service.review_architecture(request)
    except Exception as e:
        logger.exception(f"Architecture review failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def design_api(
    requirements: str,
    api_style: str = "rest",
    include_schemas: bool = True,
    include_examples: bool = True,
) -> dict[str, Any]:
    """Design API architecture and endpoints.

    Args:
        requirements: API requirements and use cases
        api_style: API style ('rest', 'graphql', 'grpc', 'websocket')
        include_schemas: Whether to include data schemas
        include_examples: Whether to include usage examples

    Returns:
        API design with endpoints and schemas

    """
    try:
        request = {
            "requirements": requirements,
            "design_type": "api",
            "api_style": api_style,
            "include_schemas": include_schemas,
            "include_examples": include_examples,
        }

        return await agent_service.design_api(request)
    except Exception as e:
        logger.exception(f"API design failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def design_database(
    requirements: str,
    database_type: str = "relational",
    include_indexes: bool = True,
    include_migrations: bool = False,
) -> dict[str, Any]:
    """Design database schema and architecture.

    Args:
        requirements: Data requirements and relationships
        database_type: Database type ('relational', 'document', 'graph', 'time-series')
        include_indexes: Whether to include index recommendations
        include_migrations: Whether to include migration scripts

    Returns:
        Database design with schemas and recommendations

    """
    try:
        request = {
            "requirements": requirements,
            "design_type": "database",
            "database_type": database_type,
            "include_indexes": include_indexes,
            "include_migrations": include_migrations,
        }

        return await agent_service.design_database(request)
    except Exception as e:
        logger.exception(f"Database design failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def plan_refactoring(
    project_path: str,
    refactoring_goals: str,
    preserve_functionality: bool = True,
    estimate_effort: bool = True,
) -> dict[str, Any]:
    """Plan a refactoring strategy for existing code.

    Args:
        project_path: Path to the project to refactor
        refactoring_goals: Comma-separated goals (e.g., 'improve-performance,reduce-complexity')
        preserve_functionality: Whether to ensure functionality is preserved
        estimate_effort: Whether to estimate refactoring effort

    Returns:
        Refactoring plan with steps and recommendations

    """
    try:
        # Parse refactoring goals from comma-separated string
        refactoring_goals_list = [
            goal.strip() for goal in refactoring_goals.split(",") if goal.strip()
        ]

        request = {
            "project_path": project_path,
            "operation": "refactoring",
            "refactoring_goals": refactoring_goals_list,
            "preserve_functionality": preserve_functionality,
            "estimate_effort": estimate_effort,
        }

        return await agent_service.plan_refactoring(request)
    except Exception as e:
        logger.exception(f"Refactoring planning failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def design_security_architecture(
    requirements: str,
    threat_model: bool = True,
    compliance_requirements: str = "",
) -> dict[str, Any]:
    """Design security architecture for a system.

    Args:
        requirements: Security requirements and constraints
        threat_model: Whether to include threat modeling
        compliance_requirements: Comma-separated compliance standards (e.g., 'GDPR,HIPAA')

    Returns:
        Security architecture design with recommendations

    """
    try:
        # Parse compliance_requirements from comma-separated string
        compliance_requirements_list = (
            [item.strip() for item in compliance_requirements.split(",") if item.strip()]
            if compliance_requirements
            else []
        )

        # Parse compliance_requirements from comma-separated string
        compliance_requirements_list = (
            [item.strip() for item in compliance_requirements.split(",") if item.strip()]
            if compliance_requirements
            else []
        )

        # Parse compliance_requirements from comma-separated string
        compliance_requirements_list = (
            [item.strip() for item in compliance_requirements.split(",") if item.strip()]
            if compliance_requirements
            else []
        )

        # Parse compliance_requirements from comma-separated string
        compliance_requirements_list = (
            [item.strip() for item in compliance_requirements.split(",") if item.strip()]
            if compliance_requirements
            else []
        )

        # Parse compliance requirements from comma-separated string
        compliance_requirements_list = (
            [req.strip() for req in compliance_requirements.split(",") if req.strip()]
            if compliance_requirements
            else []
        )

        request = {
            "requirements": requirements,
            "design_type": "security",
            "threat_model": threat_model,
            "compliance_requirements": compliance_requirements_list,
        }

        return await agent_service.design_security_architecture(request)
    except Exception as e:
        logger.exception(f"Security architecture design failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_design_status(job_id: str) -> dict[str, Any]:
    """Get the status of an ongoing design job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and progress information

    """
    try:
        return await agent_service.get_job_status(job_id)
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

"""Master Architect Agent Service - System architecture design and recommendations.
Provides comprehensive system design, architecture patterns, and technical recommendations.

Enhanced with JSON RPC 2.0 compliance for standardized request/response patterns.
"""

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.agents.rpc_parameter_models import AnalyzeArchitectureParams
from gterminal.agents.rpc_parameter_models import DesignSystemParams
from gterminal.agents.rpc_parameter_models import RecommendTechnologiesParams
from gterminal.core.rpc.models import SessionContext
from gterminal.core.rpc.patterns import RpcAgentMixin
from gterminal.core.rpc.patterns import rpc_method
from gterminal.core.security.security_utils import safe_json_parse


class MasterArchitectService(BaseAgentService, RpcAgentMixin):
    """Comprehensive system architecture design service.

    Features:
    - System architecture analysis and design
    - Technology stack recommendations
    - Scalability and performance architecture
    - Security architecture patterns
    - Microservices design patterns
    - Database architecture design
    - API design patterns
    - Infrastructure recommendations
    - Architecture documentation generation
    """

    def __init__(self) -> None:
        super().__init__(
            "master_architect",
            "System architecture design and technical recommendations",
        )

        # Architecture patterns and best practices
        self.architecture_patterns = {
            "microservices": {
                "description": "Distributed services architecture",
                "use_cases": [
                    "Large scale applications",
                    "Team autonomy",
                    "Technology diversity",
                ],
                "pros": ["Scalability", "Technology freedom", "Team independence"],
                "cons": [
                    "Complexity",
                    "Network overhead",
                    "Data consistency challenges",
                ],
            },
            "monolith": {
                "description": "Single deployable unit architecture",
                "use_cases": [
                    "Small to medium applications",
                    "Simple deployment",
                    "Team cohesion",
                ],
                "pros": ["Simplicity", "Easy deployment", "Strong consistency"],
                "cons": [
                    "Scaling limitations",
                    "Technology lock-in",
                    "Team bottlenecks",
                ],
            },
            "serverless": {
                "description": "Function-as-a-Service architecture",
                "use_cases": [
                    "Event-driven applications",
                    "Variable workloads",
                    "Rapid scaling",
                ],
                "pros": ["Auto-scaling", "Pay-per-use", "No server management"],
                "cons": ["Vendor lock-in", "Cold starts", "Limited execution time"],
            },
            "event_driven": {
                "description": "Asynchronous event-based architecture",
                "use_cases": [
                    "Real-time systems",
                    "Decoupled components",
                    "High throughput",
                ],
                "pros": ["Loose coupling", "Scalability", "Responsiveness"],
                "cons": ["Complexity", "Debugging challenges", "Eventual consistency"],
            },
        }

        # Technology recommendations by domain
        self.tech_recommendations = {
            "web_backend": {
                "python": ["FastAPI", "Django", "Flask"],
                "javascript": ["Node.js", "Express", "NestJS"],
                "java": ["Spring Boot", "Micronaut", "Quarkus"],
                "go": ["Gin", "Echo", "Fiber"],
                "rust": ["Actix-web", "Warp", "Rocket"],
            },
            "web_frontend": {
                "javascript": ["React", "Vue.js", "Angular", "Svelte"],
                "typescript": ["React", "Vue.js", "Angular"],
                "mobile": ["React Native", "Flutter", "Swift/Kotlin"],
            },
            "databases": {
                "relational": ["PostgreSQL", "MySQL", "SQLite"],
                "nosql": ["MongoDB", "Redis", "Cassandra"],
                "graph": ["Neo4j", "Amazon Neptune"],
                "time_series": ["InfluxDB", "TimescaleDB"],
            },
            "message_queues": {
                "general": ["RabbitMQ", "Apache Kafka", "Redis"],
                "cloud": ["AWS SQS", "Google Pub/Sub", "Azure Service Bus"],
            },
            "caching": {
                "in_memory": ["Redis", "Memcached"],
                "distributed": ["Hazelcast", "Apache Ignite"],
                "cdn": ["CloudFlare", "AWS CloudFront", "Azure CDN"],
            },
        }

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type == "design_system":
            return ["requirements"]
        if job_type == "analyze_architecture":
            return ["project_path"]
        if job_type == "recommend_technologies":
            return ["project_requirements"]
        if job_type == "design_api":
            return ["api_requirements"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute master architect job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "design_system":
            return await self._design_system(job, parameters["requirements"])
        if job_type == "analyze_architecture":
            return await self._analyze_architecture(job, parameters["project_path"])
        if job_type == "recommend_technologies":
            return await self._recommend_technologies(job, parameters["project_requirements"])
        if job_type == "design_api":
            return await self._design_api(job, parameters["api_requirements"])
        if job_type == "design_database":
            return await self._design_database(job, parameters["data_requirements"])
        if job_type == "security_architecture":
            return await self._design_security_architecture(
                job, parameters["security_requirements"]
            )
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    # JSON RPC 2.0 Compliant Methods

    @rpc_method(
        method_name="design_system", timeout_seconds=600, validate_params=True, log_performance=True
    )
    async def design_system_rpc(
        self,
        params: DesignSystemParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Design system architecture with RPC compliance.

        Args:
            params: System design parameters
            session: Optional session context

        Returns:
            System design result
        """
        job = Job(
            job_id=f"rpc_design_system_{datetime.now(UTC).timestamp()}",
            job_type="design_system",
            parameters={"requirements": params.requirements},
        )

        return await self._design_system(job, params.requirements)

    @rpc_method(
        method_name="recommend_technologies",
        timeout_seconds=180,
        validate_params=True,
        log_performance=True,
    )
    async def recommend_technologies_rpc(
        self,
        params: RecommendTechnologiesParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Recommend technologies for project with RPC compliance.

        Args:
            params: Technology recommendation parameters
            session: Optional session context

        Returns:
            Technology recommendations result
        """
        job = Job(
            job_id=f"rpc_recommend_tech_{datetime.now(UTC).timestamp()}",
            job_type="recommend_technologies",
            parameters={"project_type": params.project_type, "requirements": params.requirements},
        )

        return await self._recommend_technologies(job, params.project_type, params.requirements)

    @rpc_method(
        method_name="analyze_architecture",
        timeout_seconds=450,
        validate_params=True,
        log_performance=True,
    )
    async def analyze_architecture_rpc(
        self,
        params: AnalyzeArchitectureParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Analyze existing architecture with RPC compliance.

        Args:
            params: Architecture analysis parameters
            session: Optional session context

        Returns:
            Architecture analysis result
        """
        job = Job(
            job_id=f"rpc_analyze_arch_{datetime.now(UTC).timestamp()}",
            job_type="analyze_architecture",
            parameters={"project_path": params.project_path},
        )

        return await self._analyze_architecture(job, params.project_path)

    async def _design_system(self, job: Job, requirements: dict[str, Any]) -> dict[str, Any]:
        """Design comprehensive system architecture."""
        job.update_progress(10.0, "Analyzing requirements")

        system_design = {
            "requirements_analysis": {},
            "architecture_pattern": {},
            "technology_stack": {},
            "system_components": [],
            "data_flow": {},
            "scalability_plan": {},
            "security_considerations": {},
            "deployment_strategy": {},
            "monitoring_strategy": {},
            "documentation": {},
        }

        # Analyze requirements
        job.update_progress(20.0, "Analyzing functional and non-functional requirements")
        system_design["requirements_analysis"] = await self._analyze_requirements(requirements)

        # Select architecture pattern
        job.update_progress(35.0, "Selecting optimal architecture pattern")
        system_design["architecture_pattern"] = await self._select_architecture_pattern(
            system_design["requirements_analysis"],
        )

        # Design technology stack
        job.update_progress(50.0, "Designing technology stack")
        system_design["technology_stack"] = await self._design_technology_stack(
            system_design["requirements_analysis"],
            system_design["architecture_pattern"],
        )

        # Design system components
        job.update_progress(65.0, "Designing system components")
        system_design["system_components"] = await self._design_system_components(
            system_design["requirements_analysis"],
            system_design["architecture_pattern"],
        )

        # Design data flow
        job.update_progress(75.0, "Designing data flow and communication")
        system_design["data_flow"] = await self._design_data_flow(
            system_design["system_components"]
        )

        # Plan scalability
        job.update_progress(85.0, "Planning scalability strategy")
        system_design["scalability_plan"] = await self._plan_scalability(
            system_design["requirements_analysis"],
            system_design["architecture_pattern"],
        )

        # Security considerations
        job.update_progress(90.0, "Defining security architecture")
        system_design["security_considerations"] = await self._define_security_architecture(
            system_design["requirements_analysis"],
        )

        # Deployment strategy
        job.update_progress(95.0, "Planning deployment strategy")
        system_design["deployment_strategy"] = await self._plan_deployment_strategy(
            system_design["technology_stack"],
            system_design["architecture_pattern"],
        )

        job.update_progress(100.0, "System design complete")

        return {
            "requirements": requirements,
            "system_design": system_design,
            "recommendations": await self._generate_architecture_recommendations(system_design),
            "designed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _analyze_architecture(self, job: Job, project_path: str) -> dict[str, Any]:
        """Analyze existing project architecture."""
        job.update_progress(10.0, f"Starting architecture analysis of {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        analysis = {
            "current_architecture": {},
            "identified_patterns": [],
            "technology_assessment": {},
            "architectural_issues": [],
            "improvement_recommendations": [],
            "migration_suggestions": [],
        }

        # Analyze current architecture
        job.update_progress(30.0, "Analyzing current architecture")
        analysis["current_architecture"] = await self._analyze_current_architecture(project_dir)

        # Identify architectural patterns
        job.update_progress(50.0, "Identifying architectural patterns")
        analysis["identified_patterns"] = await self._identify_architectural_patterns(
            project_dir,
            analysis["current_architecture"],
        )

        # Assess technology choices
        job.update_progress(70.0, "Assessing technology choices")
        analysis["technology_assessment"] = await self._assess_technology_choices(project_dir)

        # Identify architectural issues
        job.update_progress(85.0, "Identifying architectural issues")
        analysis["architectural_issues"] = await self._identify_architectural_issues(
            analysis["current_architecture"],
            analysis["technology_assessment"],
        )

        # Generate recommendations
        job.update_progress(95.0, "Generating improvement recommendations")
        analysis["improvement_recommendations"] = await self._generate_improvement_recommendations(
            analysis["architectural_issues"],
            analysis["current_architecture"],
        )

        job.update_progress(100.0, "Architecture analysis complete")

        return {
            "project_path": project_path,
            "architecture_analysis": analysis,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _recommend_technologies(
        self, job: Job, project_requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend optimal technology stack."""
        job.update_progress(10.0, "Analyzing project requirements")

        recommendations = {
            "backend_technologies": [],
            "frontend_technologies": [],
            "database_recommendations": [],
            "infrastructure_recommendations": [],
            "development_tools": [],
            "rationale": {},
        }

        # Analyze requirements
        job.update_progress(30.0, "Matching requirements to technologies")
        req_analysis = await self._analyze_tech_requirements(project_requirements)

        # Backend recommendations
        job.update_progress(50.0, "Generating backend recommendations")
        recommendations["backend_technologies"] = await self._recommend_backend_tech(req_analysis)

        # Frontend recommendations
        job.update_progress(65.0, "Generating frontend recommendations")
        recommendations["frontend_technologies"] = await self._recommend_frontend_tech(req_analysis)

        # Database recommendations
        job.update_progress(80.0, "Generating database recommendations")
        recommendations["database_recommendations"] = await self._recommend_database_tech(
            req_analysis
        )

        # Infrastructure recommendations
        job.update_progress(90.0, "Generating infrastructure recommendations")
        recommendations[
            "infrastructure_recommendations"
        ] = await self._recommend_infrastructure_tech(req_analysis)

        # Generate rationale
        job.update_progress(95.0, "Generating recommendation rationale")
        recommendations["rationale"] = await self._generate_tech_rationale(
            recommendations, req_analysis
        )

        job.update_progress(100.0, "Technology recommendations complete")

        return {
            "project_requirements": project_requirements,
            "technology_recommendations": recommendations,
            "recommended_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _design_api(self, job: Job, api_requirements: dict[str, Any]) -> dict[str, Any]:
        """Design API architecture and endpoints."""
        job.update_progress(10.0, "Analyzing API requirements")

        api_design = {
            "api_architecture": {},
            "endpoint_design": [],
            "data_models": [],
            "authentication_strategy": {},
            "versioning_strategy": {},
            "documentation_strategy": {},
            "testing_strategy": {},
        }

        # Design API architecture
        job.update_progress(30.0, "Designing API architecture")
        api_design["api_architecture"] = await self._design_api_architecture(api_requirements)

        # Design endpoints
        job.update_progress(50.0, "Designing API endpoints")
        api_design["endpoint_design"] = await self._design_api_endpoints(api_requirements)

        # Design data models
        job.update_progress(65.0, "Designing data models")
        api_design["data_models"] = await self._design_api_data_models(api_requirements)

        # Design authentication
        job.update_progress(80.0, "Designing authentication strategy")
        api_design["authentication_strategy"] = await self._design_api_authentication(
            api_requirements
        )

        # Design versioning and documentation
        job.update_progress(90.0, "Designing versioning and documentation")
        api_design["versioning_strategy"] = await self._design_api_versioning(api_requirements)
        api_design["documentation_strategy"] = await self._design_api_documentation(
            api_requirements
        )

        job.update_progress(100.0, "API design complete")

        return {
            "api_requirements": api_requirements,
            "api_design": api_design,
            "designed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _design_database(self, job: Job, data_requirements: dict[str, Any]) -> dict[str, Any]:
        """Design database architecture."""
        job.update_progress(10.0, "Analyzing data requirements")

        db_design = {
            "database_type": "",
            "schema_design": {},
            "indexing_strategy": {},
            "scaling_strategy": {},
            "backup_strategy": {},
            "performance_optimization": {},
        }

        # Select database type
        job.update_progress(30.0, "Selecting optimal database type")
        db_design["database_type"] = await self._select_database_type(data_requirements)

        # Design schema
        job.update_progress(50.0, "Designing database schema")
        db_design["schema_design"] = await self._design_database_schema(
            data_requirements, db_design["database_type"]
        )

        # Design indexing strategy
        job.update_progress(70.0, "Designing indexing strategy")
        db_design["indexing_strategy"] = await self._design_indexing_strategy(
            db_design["schema_design"]
        )

        # Plan scaling and backup
        job.update_progress(85.0, "Planning scaling and backup strategies")
        db_design["scaling_strategy"] = await self._plan_database_scaling(data_requirements)
        db_design["backup_strategy"] = await self._plan_backup_strategy(data_requirements)

        job.update_progress(100.0, "Database design complete")

        return {
            "data_requirements": data_requirements,
            "database_design": db_design,
            "designed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _design_security_architecture(
        self, job: Job, security_requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Design comprehensive security architecture."""
        job.update_progress(10.0, "Analyzing security requirements")

        security_design = {
            "authentication_architecture": {},
            "authorization_model": {},
            "data_protection": {},
            "network_security": {},
            "monitoring_and_logging": {},
            "compliance_considerations": {},
            "threat_model": {},
        }

        # Design authentication
        job.update_progress(25.0, "Designing authentication architecture")
        security_design[
            "authentication_architecture"
        ] = await self._design_authentication_architecture(
            security_requirements,
        )

        # Design authorization
        job.update_progress(40.0, "Designing authorization model")
        security_design["authorization_model"] = await self._design_authorization_model(
            security_requirements
        )

        # Design data protection
        job.update_progress(55.0, "Designing data protection strategy")
        security_design["data_protection"] = await self._design_data_protection(
            security_requirements
        )

        # Design network security
        job.update_progress(70.0, "Designing network security")
        security_design["network_security"] = await self._design_network_security(
            security_requirements
        )

        # Design monitoring
        job.update_progress(85.0, "Designing security monitoring")
        security_design["monitoring_and_logging"] = await self._design_security_monitoring(
            security_requirements
        )

        job.update_progress(100.0, "Security architecture design complete")

        return {
            "security_requirements": security_requirements,
            "security_architecture": security_design,
            "designed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _analyze_requirements(self, requirements: dict[str, Any]) -> dict[str, Any]:
        """Analyze functional and non-functional requirements."""
        analysis = {
            "functional_requirements": [],
            "non_functional_requirements": {},
            "constraints": [],
            "assumptions": [],
            "scale_requirements": {},
            "performance_requirements": {},
            "security_requirements": {},
        }

        # Extract and categorize requirements
        if "functional" in requirements:
            analysis["functional_requirements"] = requirements["functional"]

        if "non_functional" in requirements:
            analysis["non_functional_requirements"] = requirements["non_functional"]

        # Extract specific requirement types
        analysis["scale_requirements"] = {
            "users": requirements.get("expected_users", "unknown"),
            "data_volume": requirements.get("data_volume", "unknown"),
            "transaction_volume": requirements.get("transaction_volume", "unknown"),
            "geographic_distribution": requirements.get("geographic_scope", "single_region"),
        }

        analysis["performance_requirements"] = {
            "response_time": requirements.get("response_time", "< 200ms"),
            "throughput": requirements.get("throughput", "unknown"),
            "availability": requirements.get("availability", "99.9%"),
            "consistency": requirements.get("consistency", "eventual"),
        }

        analysis["security_requirements"] = {
            "authentication": requirements.get("authentication", "required"),
            "authorization": requirements.get("authorization", "role_based"),
            "data_encryption": requirements.get("encryption", "at_rest_and_transit"),
            "compliance": requirements.get("compliance", []),
        }

        return analysis

    async def _select_architecture_pattern(
        self, requirements_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Select optimal architecture pattern based on requirements."""
        scale_req = requirements_analysis.get("scale_requirements", {})
        perf_req = requirements_analysis.get("performance_requirements", {})

        # Decision logic for architecture pattern
        user_count = scale_req.get("users", "unknown")
        geographic_scope = scale_req.get("geographic_distribution", "single_region")
        availability_req = perf_req.get("availability", "99.9%")

        selected_pattern = "monolith"  # Default
        reasoning: list[Any] = []

        # Simple decision tree (in production, this would be more sophisticated)
        if isinstance(user_count, str) and "million" in user_count.lower():
            selected_pattern = "microservices"
            reasoning.append("High user count requires microservices for scalability")
        elif geographic_scope == "global":
            selected_pattern = "microservices"
            reasoning.append("Global distribution benefits from microservices architecture")
        elif "99.99" in availability_req:
            selected_pattern = "microservices"
            reasoning.append("High availability requirements favor microservices")
        elif "event" in str(requirements_analysis.get("functional_requirements", [])).lower():
            selected_pattern = "event_driven"
            reasoning.append("Event-driven requirements identified")

        pattern_info = self.architecture_patterns.get(selected_pattern, {})

        return {
            "selected_pattern": selected_pattern,
            "reasoning": reasoning,
            "pattern_details": pattern_info,
            "alternatives": [p for p in self.architecture_patterns if p != selected_pattern],
        }

    async def _design_technology_stack(
        self,
        requirements_analysis: dict[str, Any],
        architecture_pattern: dict[str, Any],
    ) -> dict[str, Any]:
        """Design comprehensive technology stack."""
        # Use AI to help with technology selection
        prompt = f"""Design a technology stack for a system with these requirements:

Architecture Pattern: {architecture_pattern["selected_pattern"]}
Scale Requirements: {requirements_analysis.get("scale_requirements", {})}
Performance Requirements: {requirements_analysis.get("performance_requirements", {})}
Security Requirements: {requirements_analysis.get("security_requirements", {})}

Recommend specific technologies for:
1. Backend framework and language
2. Frontend framework and language
3. Database(s)
4. Message queues/communication
5. Caching layers
6. Monitoring and observability
7. Deployment and infrastructure

Provide reasoning for each choice. Return as JSON with structure:
{{
  "backend": {{"framework": "...", "language": "...", "reasoning": "..."}},
  "frontend": {{"framework": "...", "language": "...", "reasoning": "..."}},
  "database": {{"primary": "...", "cache": "...", "reasoning": "..."}},
  "infrastructure": {{"deployment": "...", "monitoring": "...", "reasoning": "..."}}
}}"""

        ai_response = await self.generate_with_gemini(prompt, "analysis")

        if ai_response:
            tech_stack = safe_json_parse(ai_response)
            if tech_stack:
                return tech_stack

        # Fallback to rule-based selection
        return self._fallback_tech_stack_selection(requirements_analysis, architecture_pattern)

    def _fallback_tech_stack_selection(
        self,
        requirements_analysis: dict[str, Any],
        architecture_pattern: dict[str, Any],
    ) -> dict[str, Any]:
        """Fallback technology stack selection using rules."""
        pattern = architecture_pattern["selected_pattern"]

        if pattern == "microservices":
            return {
                "backend": {
                    "framework": "FastAPI",
                    "language": "Python",
                    "reasoning": "FastAPI provides excellent performance and API documentation",
                },
                "frontend": {
                    "framework": "React",
                    "language": "TypeScript",
                    "reasoning": "React with TypeScript provides scalable frontend development",
                },
                "database": {
                    "primary": "PostgreSQL",
                    "cache": "Redis",
                    "reasoning": "PostgreSQL for ACID compliance, Redis for caching",
                },
                "infrastructure": {
                    "deployment": "Kubernetes",
                    "monitoring": "Prometheus + Grafana",
                    "reasoning": "Kubernetes for container orchestration, Prometheus for monitoring",
                },
            }
        return {
            "backend": {
                "framework": "Django",
                "language": "Python",
                "reasoning": "Django provides rapid development for monolithic applications",
            },
            "frontend": {
                "framework": "React",
                "language": "JavaScript",
                "reasoning": "React provides component-based UI development",
            },
            "database": {
                "primary": "PostgreSQL",
                "cache": "Redis",
                "reasoning": "PostgreSQL for reliability, Redis for session storage",
            },
            "infrastructure": {
                "deployment": "Docker + AWS ECS",
                "monitoring": "CloudWatch",
                "reasoning": "Simple deployment with managed services",
            },
        }

    async def _design_system_components(
        self,
        requirements_analysis: dict[str, Any],
        architecture_pattern: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Design system components based on requirements and architecture."""
        components: list[Any] = []
        pattern = architecture_pattern["selected_pattern"]

        if pattern == "microservices":
            # Common microservices components
            components = [
                {
                    "name": "API Gateway",
                    "type": "infrastructure",
                    "purpose": "Request routing and cross-cutting concerns",
                    "responsibilities": [
                        "Authentication",
                        "Rate limiting",
                        "Request routing",
                    ],
                    "technologies": ["Kong", "AWS API Gateway", "Envoy"],
                },
                {
                    "name": "User Service",
                    "type": "business_service",
                    "purpose": "User management and authentication",
                    "responsibilities": [
                        "User registration",
                        "Authentication",
                        "Profile management",
                    ],
                    "technologies": ["FastAPI", "PostgreSQL", "Redis"],
                },
                {
                    "name": "Business Logic Service",
                    "type": "business_service",
                    "purpose": "Core business functionality",
                    "responsibilities": [
                        "Business rules",
                        "Data processing",
                        "Workflow management",
                    ],
                    "technologies": ["FastAPI", "PostgreSQL", "Message Queue"],
                },
                {
                    "name": "Notification Service",
                    "type": "support_service",
                    "purpose": "Handle notifications and communications",
                    "responsibilities": [
                        "Email notifications",
                        "SMS",
                        "Push notifications",
                    ],
                    "technologies": ["FastAPI", "Redis", "External APIs"],
                },
            ]
        else:
            # Monolithic components
            components = [
                {
                    "name": "Web Application",
                    "type": "application",
                    "purpose": "Main application serving users",
                    "responsibilities": [
                        "User interface",
                        "Business logic",
                        "Data access",
                    ],
                    "technologies": ["Django", "PostgreSQL", "Redis"],
                },
                {
                    "name": "Background Workers",
                    "type": "background_service",
                    "purpose": "Asynchronous task processing",
                    "responsibilities": [
                        "Email sending",
                        "Report generation",
                        "Data processing",
                    ],
                    "technologies": ["Celery", "Redis", "PostgreSQL"],
                },
            ]

        return components

    async def _design_data_flow(self, system_components: list[dict[str, Any]]) -> dict[str, Any]:
        """Design data flow between system components."""
        data_flow = {
            "synchronous_flows": [],
            "asynchronous_flows": [],
            "data_storage_flows": [],
            "external_integrations": [],
        }

        # Generate data flows based on components
        for i, component in enumerate(system_components):
            for j, other_component in enumerate(system_components):
                if i != j:
                    # Determine if components should communicate
                    flow_type = self._determine_flow_type(component, other_component)
                    if flow_type:
                        flow = {
                            "from": component["name"],
                            "to": other_component["name"],
                            "type": flow_type,
                            "data_format": "JSON",
                            "protocol": ("HTTP" if flow_type == "synchronous" else "Message Queue"),
                        }

                        if flow_type == "synchronous":
                            data_flow["synchronous_flows"].append(flow)
                        else:
                            data_flow["asynchronous_flows"].append(flow)

        return data_flow

    def _determine_flow_type(
        self, component1: dict[str, Any], component2: dict[str, Any]
    ) -> str | None:
        """Determine if and how two components should communicate."""
        # Simple rules for component communication
        if component1["name"] == "API Gateway" and component2["type"] == "business_service":
            return "synchronous"
        if ("Service" in component1["name"] and "Service" in component2["name"]) or (
            component1["type"] == "application" and component2["type"] == "background_service"
        ):
            return "asynchronous"
        return None

    async def _plan_scalability(
        self,
        requirements_analysis: dict[str, Any],
        architecture_pattern: dict[str, Any],
    ) -> dict[str, Any]:
        """Plan scalability strategy."""
        scale_req = requirements_analysis.get("scale_requirements", {})

        # Generate AI-powered scalability recommendations
        prompt = f"""Design a scalability plan for a system with:

Architecture: {architecture_pattern["selected_pattern"]}
Expected Users: {scale_req.get("users", "unknown")}
Data Volume: {scale_req.get("data_volume", "unknown")}
Geographic Distribution: {scale_req.get("geographic_distribution", "single_region")}

Provide specific recommendations for:
1. Horizontal vs vertical scaling strategies
2. Database scaling approaches
3. Caching strategies
4. Load balancing configuration
5. Auto-scaling policies

Return as JSON."""

        ai_response = await self.generate_with_gemini(prompt, "analysis")

        if ai_response:
            ai_plan = safe_json_parse(ai_response)
            if ai_plan:
                return ai_plan

        # Fallback scalability plan
        return {
            "horizontal_scaling": {
                "strategy": "Container-based horizontal scaling",
                "trigger_metrics": [
                    "CPU > 70%",
                    "Memory > 80%",
                    "Request latency > 500ms",
                ],
                "scaling_policy": "Scale out by 2 instances, scale in by 1",
            },
            "database_scaling": {
                "read_replicas": "Configure read replicas for read-heavy workloads",
                "connection_pooling": "Implement connection pooling to manage database connections",
                "sharding": "Consider sharding for very large datasets",
            },
            "caching_strategy": {
                "application_cache": "Redis for session storage and frequently accessed data",
                "cdn": "CloudFront/CloudFlare for static content",
                "query_cache": "Database query result caching",
            },
        }

    async def _define_security_architecture(
        self, requirements_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Define comprehensive security architecture."""
        requirements_analysis.get("security_requirements", {})

        return {
            "authentication": {
                "method": "JWT with refresh tokens",
                "providers": ["Local auth", "OAuth2 (Google, GitHub)"],
                "mfa": "Optional TOTP-based MFA",
            },
            "authorization": {
                "model": "Role-Based Access Control (RBAC)",
                "implementation": "Middleware-based authorization checks",
            },
            "data_protection": {
                "encryption_at_rest": "AES-256 database encryption",
                "encryption_in_transit": "TLS 1.3 for all communications",
                "key_management": "Cloud-based key management service",
            },
            "network_security": {
                "firewall": "Application firewall with DDoS protection",
                "vpc_isolation": "Private subnets for backend services",
                "ssl_termination": "Load balancer SSL termination",
            },
            "monitoring": {
                "security_logging": "Centralized security event logging",
                "anomaly_detection": "ML-based anomaly detection",
                "incident_response": "Automated incident response workflows",
            },
        }

    async def _plan_deployment_strategy(
        self,
        technology_stack: dict[str, Any],
        architecture_pattern: dict[str, Any],
    ) -> dict[str, Any]:
        """Plan deployment strategy."""
        pattern = architecture_pattern["selected_pattern"]

        if pattern == "microservices":
            return {
                "containerization": "Docker containers for all services",
                "orchestration": "Kubernetes for container orchestration",
                "deployment_pattern": "Blue-green deployments with canary releases",
                "ci_cd": "GitLab CI/CD with automated testing and deployment",
                "infrastructure_as_code": "Terraform for infrastructure provisioning",
                "monitoring": "Prometheus + Grafana + ELK stack",
            }
        return {
            "containerization": "Docker container for application",
            "hosting": "Cloud platform (AWS ECS, Google Cloud Run)",
            "deployment_pattern": "Rolling deployments with health checks",
            "ci_cd": "GitHub Actions with automated testing",
            "infrastructure_as_code": "CloudFormation or Terraform",
            "monitoring": "Cloud-native monitoring (CloudWatch, Stackdriver)",
        }

    async def _generate_architecture_recommendations(
        self, system_design: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate architecture recommendations."""
        recommendations: list[Any] = []

        # Analyze the design and generate recommendations
        arch_pattern = system_design.get("architecture_pattern", {})
        tech_stack = system_design.get("technology_stack", {})

        # Pattern-specific recommendations
        if arch_pattern.get("selected_pattern") == "microservices":
            recommendations.append(
                {
                    "category": "architecture",
                    "priority": "high",
                    "title": "Implement Circuit Breaker Pattern",
                    "description": "Add circuit breakers to prevent cascade failures between services",
                },
            )

            recommendations.append(
                {
                    "category": "observability",
                    "priority": "high",
                    "title": "Distributed Tracing",
                    "description": "Implement distributed tracing with Jaeger or Zipkin",
                },
            )

        # Technology-specific recommendations
        if tech_stack.get("database", {}).get("primary") == "PostgreSQL":
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "medium",
                    "title": "Database Indexing Strategy",
                    "description": "Implement comprehensive indexing strategy for query optimization",
                },
            )

        # General recommendations
        recommendations.extend(
            [
                {
                    "category": "security",
                    "priority": "high",
                    "title": "Security Scanning Integration",
                    "description": "Integrate security scanning in CI/CD pipeline",
                },
                {
                    "category": "performance",
                    "priority": "medium",
                    "title": "Performance Testing",
                    "description": "Implement automated performance testing and monitoring",
                },
            ],
        )

        return recommendations

    # Additional helper methods for architecture analysis
    async def _analyze_current_architecture(self, project_dir: Path) -> dict[str, Any]:
        """Analyze current project architecture."""
        analysis = {
            "structure_type": "unknown",
            "technology_stack": {},
            "component_count": 0,
            "complexity_indicators": {},
        }

        # Detect project structure
        if (project_dir / "services").exists() or (project_dir / "microservices").exists():
            analysis["structure_type"] = "microservices"
        elif (
            len(list(project_dir.glob("**/app.py"))) > 0
            or len(list(project_dir.glob("**/main.py"))) > 0
        ):
            analysis["structure_type"] = "monolith"

        # Count components/modules
        python_files = list(project_dir.rglob("*.py"))
        analysis["component_count"] = len(python_files)

        # Detect technologies
        if (project_dir / "requirements.txt").exists():
            analysis["technology_stack"]["backend"] = "Python"
        if (project_dir / "package.json").exists():
            analysis["technology_stack"]["frontend"] = "Node.js"

        return analysis

    async def _identify_architectural_patterns(
        self, project_dir: Path, current_arch: dict[str, Any]
    ) -> list[str]:
        """Identify architectural patterns in use."""
        patterns: list[Any] = []

        # Check for common patterns
        if (project_dir / "api" / "gateway").exists():
            patterns.append("API Gateway Pattern")

        if len(list(project_dir.rglob("*service*.py"))) > 3:
            patterns.append("Service-Oriented Architecture")

        if (project_dir / "events").exists() or "event" in str(project_dir).lower():
            patterns.append("Event-Driven Architecture")

        return patterns

    async def _assess_technology_choices(self, project_dir: Path) -> dict[str, Any]:
        """Assess current technology choices."""
        assessment = {
            "strengths": [],
            "weaknesses": [],
            "modernization_opportunities": [],
            "risk_factors": [],
        }

        # Analyze technology files
        if (project_dir / "requirements.txt").exists():
            req_content = self.safe_file_read(project_dir / "requirements.txt")
            if req_content:
                if "django==1" in req_content.lower():
                    assessment["risk_factors"].append("Outdated Django version")
                if "flask" in req_content.lower():
                    assessment["strengths"].append("Lightweight Flask framework")

        return assessment

    async def _identify_architectural_issues(
        self,
        current_arch: dict[str, Any],
        tech_assessment: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Identify architectural issues."""
        issues: list[Any] = []

        # Check for common issues
        if current_arch.get("component_count", 0) > 100:
            issues.append(
                {
                    "severity": "medium",
                    "category": "complexity",
                    "description": "High number of components may indicate complexity issues",
                    "recommendation": "Consider modularization or service extraction",
                },
            )

        if len(tech_assessment.get("risk_factors", [])) > 0:
            issues.append(
                {
                    "severity": "high",
                    "category": "technology_debt",
                    "description": "Outdated technology dependencies detected",
                    "recommendation": "Plan technology upgrade roadmap",
                },
            )

        return issues

    async def _generate_improvement_recommendations(
        self,
        issues: list[dict[str, Any]],
        current_arch: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate improvement recommendations."""
        recommendations: list[Any] = []

        for issue in issues:
            if issue["category"] == "complexity":
                recommendations.append(
                    {
                        "priority": "medium",
                        "title": "Reduce System Complexity",
                        "description": "Break down large components into smaller, focused modules",
                        "effort_estimate": "4-6 weeks",
                        "impact": "Improved maintainability and developer productivity",
                    },
                )

        # Always recommend good practices
        recommendations.extend(
            [
                {
                    "priority": "high",
                    "title": "Implement Comprehensive Monitoring",
                    "description": "Add application monitoring, logging, and alerting",
                    "effort_estimate": "2-3 weeks",
                    "impact": "Improved system reliability and faster issue resolution",
                },
                {
                    "priority": "medium",
                    "title": "Automated Testing Strategy",
                    "description": "Implement unit, integration, and end-to-end testing",
                    "effort_estimate": "3-4 weeks",
                    "impact": "Reduced bugs and improved deployment confidence",
                },
            ],
        )

        return recommendations

    def register_tools(self) -> None:
        """Register MCP tools for master architect."""

        @self.mcp.tool()
        async def design_system(requirements: dict[str, Any]) -> dict[str, Any]:
            """Design comprehensive system architecture."""
            if not self.validate_job_parameters("design_system", {"requirements": requirements}):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("design_system", {"requirements": requirements})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def analyze_architecture(project_path: str = ".") -> dict[str, Any]:
            """Analyze existing project architecture."""
            if not self.validate_job_parameters(
                "analyze_architecture", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("analyze_architecture", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def recommend_technologies(
            project_requirements: dict[str, Any],
        ) -> dict[str, Any]:
            """Recommend optimal technology stack."""
            if not self.validate_job_parameters(
                "recommend_technologies",
                {"project_requirements": project_requirements},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job(
                "recommend_technologies", {"project_requirements": project_requirements}
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def design_api(api_requirements: dict[str, Any]) -> dict[str, Any]:
            """Design API architecture and endpoints."""
            if not self.validate_job_parameters(
                "design_api", {"api_requirements": api_requirements}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("design_api", {"api_requirements": api_requirements})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_architecture_patterns() -> dict[str, Any]:
            """Get available architecture patterns and their details."""
            return self.create_success_response(
                {
                    "patterns": self.architecture_patterns,
                    "technology_recommendations": self.tech_recommendations,
                    "agent_stats": self.get_agent_stats(),
                },
            )


# Create global instance
master_architect_service = MasterArchitectService()

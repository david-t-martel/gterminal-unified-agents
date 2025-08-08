"""Comprehensive tests for cost optimization components.

This module tests cost analysis, optimization recommendations, monitoring,
and all cost-related functionality.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from unittest.mock import patch

from app.cost_optimization.cost_analyzer import CostAlert
from app.cost_optimization.cost_analyzer import CostAnalyzer
from app.cost_optimization.cost_analyzer import CostMetric
from app.cost_optimization.cost_analyzer import ResourceUsage
from app.cost_optimization.optimizer import CostOptimizer
from app.cost_optimization.optimizer import OptimizationImpact
from app.cost_optimization.optimizer import OptimizationRecommendation
import pytest


class TestCostMetric:
    """Test CostMetric data model."""

    def test_cost_metric_creation_valid(self):
        """Test valid cost metric creation."""
        metric = CostMetric(
            resource_id="compute-instance-123",
            resource_type="compute",
            cost_amount=Decimal("45.67"),
            currency="USD",
            billing_period_start=datetime.now(UTC) - timedelta(days=1),
            billing_period_end=datetime.now(UTC),
            usage_quantity=100.5,
            usage_unit="hours",
        )

        assert metric.resource_id == "compute-instance-123"
        assert metric.resource_type == "compute"
        assert metric.cost_amount == Decimal("45.67")
        assert metric.currency == "USD"
        assert isinstance(metric.billing_period_start, datetime)
        assert isinstance(metric.billing_period_end, datetime)
        assert metric.usage_quantity == 100.5
        assert metric.usage_unit == "hours"

    def test_cost_metric_validation_negative_cost(self):
        """Test validation fails with negative cost amount."""
        with pytest.raises(ValueError, match="Cost amount cannot be negative"):
            CostMetric(
                resource_id="test",
                resource_type="compute",
                cost_amount=Decimal("-10.00"),
                currency="USD",
                billing_period_start=datetime.now(UTC) - timedelta(days=1),
                billing_period_end=datetime.now(UTC),
            )

    def test_cost_metric_validation_invalid_currency(self):
        """Test validation fails with invalid currency."""
        with pytest.raises(ValueError, match="Invalid currency code"):
            CostMetric(
                resource_id="test",
                resource_type="compute",
                cost_amount=Decimal("10.00"),
                currency="INVALID",
                billing_period_start=datetime.now(UTC) - timedelta(days=1),
                billing_period_end=datetime.now(UTC),
            )

    def test_cost_metric_validation_invalid_date_range(self):
        """Test validation fails with invalid date range."""
        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="Billing period end must be after start"):
            CostMetric(
                resource_id="test",
                resource_type="compute",
                cost_amount=Decimal("10.00"),
                currency="USD",
                billing_period_start=now,
                billing_period_end=now - timedelta(hours=1),
            )

    def test_cost_metric_daily_cost_calculation(self):
        """Test daily cost calculation."""
        start_time = datetime.now(UTC) - timedelta(days=2)
        end_time = datetime.now(UTC)

        metric = CostMetric(
            resource_id="test",
            resource_type="compute",
            cost_amount=Decimal("100.00"),
            currency="USD",
            billing_period_start=start_time,
            billing_period_end=end_time,
        )

        daily_cost = metric.get_daily_cost()
        assert daily_cost == Decimal("50.00")  # $100 over 2 days = $50/day

    def test_cost_metric_hourly_cost_calculation(self):
        """Test hourly cost calculation."""
        start_time = datetime.now(UTC) - timedelta(hours=24)
        end_time = datetime.now(UTC)

        metric = CostMetric(
            resource_id="test",
            resource_type="compute",
            cost_amount=Decimal("48.00"),
            currency="USD",
            billing_period_start=start_time,
            billing_period_end=end_time,
        )

        hourly_cost = metric.get_hourly_cost()
        assert hourly_cost == Decimal("2.00")  # $48 over 24 hours = $2/hour


class TestResourceUsage:
    """Test ResourceUsage data model."""

    def test_resource_usage_creation(self):
        """Test resource usage creation."""
        usage = ResourceUsage(
            resource_id="vm-123",
            resource_type="compute",
            cpu_utilization=75.5,
            memory_utilization=60.2,
            storage_utilization=45.0,
            network_ingress_gb=12.5,
            network_egress_gb=8.3,
            measurement_time=datetime.now(UTC),
        )

        assert usage.resource_id == "vm-123"
        assert usage.cpu_utilization == 75.5
        assert usage.memory_utilization == 60.2
        assert usage.storage_utilization == 45.0
        assert usage.network_ingress_gb == 12.5
        assert usage.network_egress_gb == 8.3

    def test_resource_usage_validation_invalid_utilization(self):
        """Test validation fails with invalid utilization percentages."""
        with pytest.raises(ValueError, match="CPU utilization must be between 0 and 100"):
            ResourceUsage(
                resource_id="test",
                resource_type="compute",
                cpu_utilization=150.0,  # Invalid: > 100%
                measurement_time=datetime.now(UTC),
            )

    def test_resource_usage_validation_negative_network(self):
        """Test validation fails with negative network usage."""
        with pytest.raises(ValueError, match="Network usage cannot be negative"):
            ResourceUsage(
                resource_id="test",
                resource_type="compute",
                cpu_utilization=50.0,
                network_ingress_gb=-5.0,  # Invalid: negative
                measurement_time=datetime.now(UTC),
            )

    def test_resource_usage_efficiency_score(self):
        """Test efficiency score calculation."""
        usage = ResourceUsage(
            resource_id="test",
            resource_type="compute",
            cpu_utilization=80.0,
            memory_utilization=70.0,
            storage_utilization=60.0,
            measurement_time=datetime.now(UTC),
        )

        efficiency = usage.get_efficiency_score()
        expected = (80.0 + 70.0 + 60.0) / 3  # Average utilization
        assert efficiency == expected

    def test_resource_usage_is_underutilized(self):
        """Test underutilization detection."""
        # Low utilization resource
        low_usage = ResourceUsage(
            resource_id="test",
            resource_type="compute",
            cpu_utilization=15.0,
            memory_utilization=20.0,
            storage_utilization=10.0,
            measurement_time=datetime.now(UTC),
        )

        assert low_usage.is_underutilized(threshold=30.0) is True

        # High utilization resource
        high_usage = ResourceUsage(
            resource_id="test",
            resource_type="compute",
            cpu_utilization=85.0,
            memory_utilization=90.0,
            storage_utilization=75.0,
            measurement_time=datetime.now(UTC),
        )

        assert high_usage.is_underutilized(threshold=30.0) is False


class TestCostAlert:
    """Test CostAlert data model."""

    def test_cost_alert_creation(self):
        """Test cost alert creation."""
        alert = CostAlert(
            alert_id="alert-123",
            alert_type="budget_exceeded",
            severity="high",
            resource_id="project-abc",
            message="Monthly budget exceeded by 25%",
            threshold_value=Decimal("1000.00"),
            actual_value=Decimal("1250.00"),
            created_at=datetime.now(UTC),
        )

        assert alert.alert_id == "alert-123"
        assert alert.alert_type == "budget_exceeded"
        assert alert.severity == "high"
        assert alert.resource_id == "project-abc"
        assert alert.threshold_value == Decimal("1000.00")
        assert alert.actual_value == Decimal("1250.00")

    def test_cost_alert_validation_invalid_severity(self):
        """Test validation fails with invalid severity."""
        with pytest.raises(ValueError, match="Invalid severity level"):
            CostAlert(
                alert_id="test",
                alert_type="budget_exceeded",
                severity="invalid",  # Must be low, medium, high, or critical
                resource_id="test",
                message="Test alert",
                threshold_value=Decimal("100"),
                actual_value=Decimal("150"),
                created_at=datetime.now(UTC),
            )

    def test_cost_alert_percentage_over_threshold(self):
        """Test percentage over threshold calculation."""
        alert = CostAlert(
            alert_id="test",
            alert_type="budget_exceeded",
            severity="medium",
            resource_id="test",
            message="Test alert",
            threshold_value=Decimal("100.00"),
            actual_value=Decimal("125.00"),
            created_at=datetime.now(UTC),
        )

        percentage = alert.get_percentage_over_threshold()
        assert percentage == 25.0  # 25% over threshold


class TestCostAnalyzer:
    """Test CostAnalyzer class."""

    @pytest.fixture
    def mock_gcp_client(self):
        """Create mock GCP billing client."""
        return Mock()

    @pytest.fixture
    def cost_analyzer(self, mock_gcp_client):
        """Create CostAnalyzer with mock client."""
        return CostAnalyzer(billing_client=mock_gcp_client, project_id="test-project")

    @pytest.mark.asyncio
    async def test_analyze_project_costs_success(self, cost_analyzer, mock_gcp_client):
        """Test successful project cost analysis."""
        # Mock billing API response
        mock_response = {
            "rows": [
                {
                    "usage_start_time": "2024-01-01T00:00:00Z",
                    "usage_end_time": "2024-01-02T00:00:00Z",
                    "cost": {"currency_code": "USD", "units": "45", "nanos": 670000000},
                    "service": {"description": "Compute Engine"},
                    "sku": {"description": "VM Instance"},
                    "usage": {"amount": 24.0, "unit": "hour"},
                }
            ]
        }

        mock_gcp_client.query_billing_data.return_value = mock_response

        result = await cost_analyzer.analyze_project_costs(
            start_date=datetime.now(UTC) - timedelta(days=7), end_date=datetime.now(UTC)
        )

        assert result["status"] == "success"
        assert "total_cost" in result
        assert "cost_breakdown" in result
        assert len(result["cost_metrics"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_resource_usage_success(self, cost_analyzer):
        """Test successful resource usage analysis."""
        # Mock monitoring API response
        with patch("app.cost_optimization.cost_analyzer.monitoring_v3") as mock_monitoring:
            mock_client = Mock()
            mock_monitoring.MetricServiceClient.return_value = mock_client

            # Mock time series data
            mock_time_series = [
                Mock(
                    resource=Mock(labels={"instance_id": "vm-123"}),
                    points=[Mock(value=Mock(double_value=75.5))],
                )
            ]
            mock_client.list_time_series.return_value = mock_time_series

            result = await cost_analyzer.analyze_resource_usage("compute-instance-123")

            assert result["status"] == "success"
            assert "cpu_utilization" in result
            assert "efficiency_score" in result

    @pytest.mark.asyncio
    async def test_generate_cost_forecast_success(self, cost_analyzer):
        """Test cost forecasting."""
        # Mock historical cost data
        historical_costs = [
            CostMetric(
                resource_id="test",
                resource_type="compute",
                cost_amount=Decimal("100.00"),
                currency="USD",
                billing_period_start=datetime.now(UTC) - timedelta(days=i + 1),
                billing_period_end=datetime.now(UTC) - timedelta(days=i),
            )
            for i in range(7)  # 7 days of data
        ]

        with patch.object(cost_analyzer, "get_historical_costs", return_value=historical_costs):
            forecast = await cost_analyzer.generate_cost_forecast(forecast_days=30)

            assert forecast["status"] == "success"
            assert "forecast_total" in forecast
            assert "daily_forecast" in forecast
            assert len(forecast["daily_forecast"]) == 30

    @pytest.mark.asyncio
    async def test_detect_cost_anomalies_success(self, cost_analyzer):
        """Test cost anomaly detection."""
        # Mock cost data with anomaly
        daily_costs = [
            {"date": "2024-01-01", "cost": 100.0},
            {"date": "2024-01-02", "cost": 105.0},
            {"date": "2024-01-03", "cost": 98.0},
            {"date": "2024-01-04", "cost": 500.0},  # Anomaly
            {"date": "2024-01-05", "cost": 102.0},
        ]

        with patch.object(cost_analyzer, "get_daily_costs", return_value=daily_costs):
            anomalies = await cost_analyzer.detect_cost_anomalies(
                threshold_std_devs=2.0, min_cost_change=50.0
            )

            assert len(anomalies) > 0
            anomaly = anomalies[0]
            assert anomaly["date"] == "2024-01-04"
            assert anomaly["cost"] == 500.0
            assert anomaly["is_anomaly"] is True

    @pytest.mark.asyncio
    async def test_create_cost_alert_success(self, cost_analyzer):
        """Test cost alert creation."""
        alert = await cost_analyzer.create_cost_alert(
            alert_type="budget_exceeded",
            resource_id="project-test",
            message="Budget exceeded",
            threshold_value=Decimal("1000.00"),
            actual_value=Decimal("1200.00"),
        )

        assert isinstance(alert, CostAlert)
        assert alert.alert_type == "budget_exceeded"
        assert alert.resource_id == "project-test"
        assert alert.threshold_value == Decimal("1000.00")
        assert alert.actual_value == Decimal("1200.00")
        assert alert.severity in ["low", "medium", "high", "critical"]


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation data model."""

    def test_recommendation_creation(self):
        """Test optimization recommendation creation."""
        recommendation = OptimizationRecommendation(
            recommendation_id="rec-123",
            resource_id="vm-456",
            resource_type="compute",
            optimization_type="rightsizing",
            description="Downsize instance to n1-standard-2",
            estimated_savings=Decimal("150.00"),
            savings_currency="USD",
            confidence_score=0.85,
            implementation_effort="low",
            created_at=datetime.now(UTC),
        )

        assert recommendation.recommendation_id == "rec-123"
        assert recommendation.resource_id == "vm-456"
        assert recommendation.optimization_type == "rightsizing"
        assert recommendation.estimated_savings == Decimal("150.00")
        assert recommendation.confidence_score == 0.85
        assert recommendation.implementation_effort == "low"

    def test_recommendation_validation_invalid_confidence(self):
        """Test validation fails with invalid confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            OptimizationRecommendation(
                recommendation_id="test",
                resource_id="test",
                resource_type="compute",
                optimization_type="rightsizing",
                description="Test",
                estimated_savings=Decimal("100.00"),
                savings_currency="USD",
                confidence_score=1.5,  # Invalid: > 1.0
                implementation_effort="low",
                created_at=datetime.now(UTC),
            )

    def test_recommendation_validation_invalid_effort(self):
        """Test validation fails with invalid implementation effort."""
        with pytest.raises(ValueError, match="Invalid implementation effort"):
            OptimizationRecommendation(
                recommendation_id="test",
                resource_id="test",
                resource_type="compute",
                optimization_type="rightsizing",
                description="Test",
                estimated_savings=Decimal("100.00"),
                savings_currency="USD",
                confidence_score=0.8,
                implementation_effort="invalid",  # Must be low, medium, high
                created_at=datetime.now(UTC),
            )

    def test_recommendation_annual_savings(self):
        """Test annual savings calculation."""
        recommendation = OptimizationRecommendation(
            recommendation_id="test",
            resource_id="test",
            resource_type="compute",
            optimization_type="rightsizing",
            description="Test",
            estimated_savings=Decimal("50.00"),  # Monthly savings
            savings_currency="USD",
            confidence_score=0.8,
            implementation_effort="low",
            created_at=datetime.now(UTC),
        )

        annual_savings = recommendation.get_annual_savings()
        assert annual_savings == Decimal("600.00")  # $50 * 12 months


class TestCostOptimizer:
    """Test CostOptimizer class."""

    @pytest.fixture
    def cost_optimizer(self):
        """Create CostOptimizer instance."""
        return CostOptimizer(project_id="test-project")

    @pytest.mark.asyncio
    async def test_generate_rightsizing_recommendations(self, cost_optimizer):
        """Test rightsizing recommendation generation."""
        # Mock resource usage data
        usage_data = [
            ResourceUsage(
                resource_id="vm-underutilized",
                resource_type="compute",
                cpu_utilization=15.0,  # Low CPU usage
                memory_utilization=20.0,  # Low memory usage
                storage_utilization=30.0,
                measurement_time=datetime.now(UTC),
            ),
            ResourceUsage(
                resource_id="vm-well-utilized",
                resource_type="compute",
                cpu_utilization=75.0,  # Good CPU usage
                memory_utilization=80.0,  # Good memory usage
                storage_utilization=60.0,
                measurement_time=datetime.now(UTC),
            ),
        ]

        with patch.object(cost_optimizer, "get_resource_usage_data", return_value=usage_data):
            recommendations = await cost_optimizer.generate_rightsizing_recommendations()

            # Should recommend rightsizing for underutilized VM
            underutilized_recs = [r for r in recommendations if r.resource_id == "vm-underutilized"]
            assert len(underutilized_recs) > 0

            # Should not recommend rightsizing for well-utilized VM
            well_utilized_recs = [r for r in recommendations if r.resource_id == "vm-well-utilized"]
            assert len(well_utilized_recs) == 0

    @pytest.mark.asyncio
    async def test_generate_scheduling_recommendations(self, cost_optimizer):
        """Test scheduling recommendation generation."""
        # Mock cost data showing usage patterns
        with patch.object(cost_optimizer, "analyze_usage_patterns") as mock_analyze:
            mock_analyze.return_value = {
                "vm-dev-instance": {
                    "usage_pattern": "business_hours_only",
                    "off_hours_percentage": 65.0,
                    "potential_savings": Decimal("200.00"),
                }
            }

            recommendations = await cost_optimizer.generate_scheduling_recommendations()

            assert len(recommendations) > 0
            rec = recommendations[0]
            assert rec.optimization_type == "scheduling"
            assert rec.resource_id == "vm-dev-instance"
            assert rec.estimated_savings == Decimal("200.00")

    @pytest.mark.asyncio
    async def test_generate_storage_optimization_recommendations(self, cost_optimizer):
        """Test storage optimization recommendations."""
        # Mock storage analysis data
        with patch.object(cost_optimizer, "analyze_storage_usage") as mock_analyze:
            mock_analyze.return_value = [
                {
                    "resource_id": "disk-unused",
                    "storage_type": "persistent_disk",
                    "size_gb": 500,
                    "used_gb": 50,
                    "utilization_percentage": 10.0,
                    "monthly_cost": Decimal("50.00"),
                }
            ]

            recommendations = await cost_optimizer.generate_storage_optimization_recommendations()

            assert len(recommendations) > 0
            rec = recommendations[0]
            assert rec.optimization_type == "storage_optimization"
            assert rec.resource_id == "disk-unused"
            assert rec.estimated_savings > Decimal("0")

    @pytest.mark.asyncio
    async def test_prioritize_recommendations(self, cost_optimizer):
        """Test recommendation prioritization."""
        recommendations = [
            OptimizationRecommendation(
                recommendation_id="rec-1",
                resource_id="resource-1",
                resource_type="compute",
                optimization_type="rightsizing",
                description="Low impact",
                estimated_savings=Decimal("50.00"),
                savings_currency="USD",
                confidence_score=0.6,
                implementation_effort="high",
                created_at=datetime.now(UTC),
            ),
            OptimizationRecommendation(
                recommendation_id="rec-2",
                resource_id="resource-2",
                resource_type="compute",
                optimization_type="rightsizing",
                description="High impact",
                estimated_savings=Decimal("500.00"),
                savings_currency="USD",
                confidence_score=0.9,
                implementation_effort="low",
                created_at=datetime.now(UTC),
            ),
        ]

        prioritized = await cost_optimizer.prioritize_recommendations(recommendations)

        # High-impact, low-effort recommendation should be first
        assert prioritized[0].recommendation_id == "rec-2"
        assert prioritized[1].recommendation_id == "rec-1"

    @pytest.mark.asyncio
    async def test_calculate_optimization_impact(self, cost_optimizer):
        """Test optimization impact calculation."""
        recommendations = [
            OptimizationRecommendation(
                recommendation_id="rec-1",
                resource_id="resource-1",
                resource_type="compute",
                optimization_type="rightsizing",
                description="Rightsize VM",
                estimated_savings=Decimal("200.00"),
                savings_currency="USD",
                confidence_score=0.8,
                implementation_effort="low",
                created_at=datetime.now(UTC),
            ),
            OptimizationRecommendation(
                recommendation_id="rec-2",
                resource_id="resource-2",
                resource_type="storage",
                optimization_type="storage_optimization",
                description="Remove unused disk",
                estimated_savings=Decimal("100.00"),
                savings_currency="USD",
                confidence_score=0.95,
                implementation_effort="low",
                created_at=datetime.now(UTC),
            ),
        ]

        impact = await cost_optimizer.calculate_optimization_impact(recommendations)

        assert impact.total_monthly_savings == Decimal("300.00")
        assert impact.total_annual_savings == Decimal("3600.00")
        assert impact.number_of_recommendations == 2
        assert "rightsizing" in impact.optimization_breakdown
        assert "storage_optimization" in impact.optimization_breakdown


class TestOptimizationImpact:
    """Test OptimizationImpact data model."""

    def test_optimization_impact_creation(self):
        """Test optimization impact creation."""
        impact = OptimizationImpact(
            total_monthly_savings=Decimal("500.00"),
            total_annual_savings=Decimal("6000.00"),
            number_of_recommendations=5,
            optimization_breakdown={
                "rightsizing": Decimal("300.00"),
                "scheduling": Decimal("150.00"),
                "storage_optimization": Decimal("50.00"),
            },
            average_confidence_score=0.85,
            implementation_effort_distribution={"low": 3, "medium": 1, "high": 1},
        )

        assert impact.total_monthly_savings == Decimal("500.00")
        assert impact.total_annual_savings == Decimal("6000.00")
        assert impact.number_of_recommendations == 5
        assert impact.average_confidence_score == 0.85
        assert sum(impact.implementation_effort_distribution.values()) == 5

    def test_optimization_impact_percentage_by_type(self):
        """Test optimization percentage breakdown by type."""
        impact = OptimizationImpact(
            total_monthly_savings=Decimal("400.00"),
            total_annual_savings=Decimal("4800.00"),
            number_of_recommendations=3,
            optimization_breakdown={
                "rightsizing": Decimal("200.00"),  # 50%
                "scheduling": Decimal("120.00"),  # 30%
                "storage_optimization": Decimal("80.00"),  # 20%
            },
            average_confidence_score=0.8,
            implementation_effort_distribution={"low": 3},
        )

        percentages = impact.get_optimization_percentages()

        assert percentages["rightsizing"] == 50.0
        assert percentages["scheduling"] == 30.0
        assert percentages["storage_optimization"] == 20.0


class TestCostOptimizationIntegration:
    """Test cost optimization component integration."""

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self):
        """Test complete cost optimization workflow."""
        # Create analyzer and optimizer
        cost_analyzer = CostAnalyzer(billing_client=Mock(), project_id="test-project")
        cost_optimizer = CostOptimizer(project_id="test-project")

        # Mock cost analysis results
        with patch.object(cost_analyzer, "analyze_project_costs") as mock_analyze:
            mock_analyze.return_value = {
                "status": "success",
                "total_cost": Decimal("1500.00"),
                "cost_breakdown": {"compute": Decimal("1000.00"), "storage": Decimal("500.00")},
            }

            # Mock optimization recommendations
            with patch.object(cost_optimizer, "generate_all_recommendations") as mock_optimize:
                mock_recommendations = [
                    OptimizationRecommendation(
                        recommendation_id="rec-1",
                        resource_id="vm-1",
                        resource_type="compute",
                        optimization_type="rightsizing",
                        description="Downsize VM",
                        estimated_savings=Decimal("300.00"),
                        savings_currency="USD",
                        confidence_score=0.9,
                        implementation_effort="low",
                        created_at=datetime.now(UTC),
                    )
                ]
                mock_optimize.return_value = mock_recommendations

                # Run analysis
                cost_analysis = await cost_analyzer.analyze_project_costs(
                    start_date=datetime.now(UTC) - timedelta(days=30), end_date=datetime.now(UTC)
                )

                # Generate recommendations
                recommendations = await cost_optimizer.generate_all_recommendations()

                # Calculate impact
                impact = await cost_optimizer.calculate_optimization_impact(recommendations)

                # Verify workflow results
                assert cost_analysis["status"] == "success"
                assert len(recommendations) == 1
                assert impact.total_monthly_savings == Decimal("300.00")
                assert impact.number_of_recommendations == 1

    @pytest.mark.asyncio
    async def test_cost_monitoring_and_alerting(self):
        """Test cost monitoring and alerting workflow."""
        cost_analyzer = CostAnalyzer(billing_client=Mock(), project_id="test-project")

        # Mock anomaly detection
        with patch.object(cost_analyzer, "detect_cost_anomalies") as mock_detect:
            mock_detect.return_value = [
                {
                    "date": "2024-01-15",
                    "cost": 2000.0,
                    "expected_cost": 1000.0,
                    "is_anomaly": True,
                    "severity": "high",
                }
            ]

            # Detect anomalies
            anomalies = await cost_analyzer.detect_cost_anomalies()

            # Create alert for anomaly
            if anomalies:
                anomaly = anomalies[0]
                alert = await cost_analyzer.create_cost_alert(
                    alert_type="cost_spike",
                    resource_id="test-project",
                    message=f"Cost spike detected on {anomaly['date']}",
                    threshold_value=Decimal(str(anomaly["expected_cost"])),
                    actual_value=Decimal(str(anomaly["cost"])),
                )

                assert isinstance(alert, CostAlert)
                assert alert.alert_type == "cost_spike"
                assert alert.actual_value == Decimal("2000.0")
                assert alert.threshold_value == Decimal("1000.0")
                assert alert.get_percentage_over_threshold() == 100.0

from .visualization import create_feature_importance_chart, create_risk_comparison_bar, create_risk_gauge


def generate_pdf_report(*args, **kwargs):
    from .pdf_generator import generate_pdf_report as _generate_pdf_report

    return _generate_pdf_report(*args, **kwargs)

__all__ = [
    "create_feature_importance_chart",
    "create_risk_comparison_bar",
    "create_risk_gauge",
    "generate_pdf_report",
]

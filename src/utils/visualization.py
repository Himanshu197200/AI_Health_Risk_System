import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_risk_gauge(disease_name, risk_score, risk_category):
    color_map = {"Low": "#2f855a", "Medium": "#dd6b20", "High": "#c53030"}
    bar_color = color_map.get(risk_category, "#4a5568")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": disease_name},
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 33], "color": "#e6fffa"},
                    {"range": [33, 66], "color": "#feebc8"},
                    {"range": [66, 100], "color": "#fed7d7"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


def create_risk_comparison_bar(predictions):
    rows = []
    for disease, result in predictions.items():
        if "error" in result:
            continue
        rows.append(
            {
                "Disease": disease.replace("_", " ").title(),
                "Risk Score": result["risk_score"],
                "Category": result["risk_category"],
            }
        )

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Disease",
        y="Risk Score",
        color="Category",
        text="Risk Score",
        color_discrete_map={"Low": "#2f855a", "Medium": "#dd6b20", "High": "#c53030"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=380, yaxis_range=[0, 100], margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def create_feature_importance_chart(feature_importance, top_n=5):
    items = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n]
    df = pd.DataFrame(items, columns=["Feature", "Impact"])
    fig = px.bar(df, x="Impact", y="Feature", orientation="h", text="Impact", color="Impact", color_continuous_scale="YlOrRd")
    fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 30, "b": 20}, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig

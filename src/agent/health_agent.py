from datetime import datetime

from src.agent.knowledge_base import MedicalKnowledgeBase
from src.agent.llm_client import LLMClient


DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "hypertension": "Hypertension",
    "obesity": "Obesity",
    "cholesterol": "Cholesterol",
}


class HealthAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.knowledge_base = MedicalKnowledgeBase()

    def analyze_patient(self, predictions, patient_data, user_query="", memory=None):
        state = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query.strip(),
            "predictions": predictions,
            "patient_data": patient_data,
            "memory": memory or [],
        }
        state["priority_risks"] = self._identify_priority_risks(predictions)
        state["knowledge"] = self.knowledge_base.retrieve(
            diseases=[item["disease"] for item in state["priority_risks"]],
            user_query=state["user_query"],
            top_k=4,
        )

        if self.llm.available:
            try:
                report = self._generate_llm_report(state)
                generation_mode = f"llm:{self.llm.provider}"
            except Exception:
                report = self._generate_rule_based_report(state)
                generation_mode = "rule_based_fallback"
        else:
            report = self._generate_rule_based_report(state)
            generation_mode = "rule_based"

        return {
            "report": report,
            "high_risk_areas": [item["disease"] for item in state["priority_risks"]],
            "timestamp": state["timestamp"],
            "generation_mode": generation_mode,
            "agent_state": {
                "priority_risks": state["priority_risks"],
                "knowledge_topics": [item["topic"] for item in state["knowledge"]],
                "query_used": state["user_query"],
            },
            "follow_up_questions": self._generate_follow_up_questions(state),
        }

    def answer_question(self, predictions, patient_data, user_query="", memory=None):
        state = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query.strip(),
            "predictions": predictions,
            "patient_data": patient_data,
            "memory": memory or [],
        }
        state["priority_risks"] = self._identify_priority_risks(predictions)
        state["knowledge"] = self.knowledge_base.retrieve(
            diseases=[item["disease"] for item in state["priority_risks"]],
            user_query=state["user_query"],
            top_k=4,
        )

        answer_text = self._direct_answer_to_query(
            state["user_query"],
            state["priority_risks"],
            state["knowledge"],
        )
        evidence = self._build_answer_evidence(state["knowledge"])

        return {
            "question": state["user_query"] or "Follow-up question",
            "answer": answer_text,
            "evidence": evidence,
            "timestamp": state["timestamp"],
            "priority_areas": [item["disease"] for item in state["priority_risks"]],
        }

    def _identify_priority_risks(self, predictions):
        ranked = []
        for disease, result in predictions.items():
            if "error" in result:
                continue
            ranked.append(
                {
                    "disease": disease,
                    "risk_score": result.get("risk_score", 0.0),
                    "risk_category": result.get("risk_category", result.get("risk_level", "Unknown")),
                    "quality": result.get("model_quality", {}),
                }
            )
        ranked.sort(key=lambda item: item["risk_score"], reverse=True)
        return [item for item in ranked if item["risk_category"] in {"Medium", "High"}][:3]

    def _generate_llm_report(self, state):
        prompt = self._build_prompt(state)
        return self.llm.generate(prompt)

    def _build_prompt(self, state):
        lines = [
            "Create a structured markdown health report.",
            "Use clear patient-friendly language.",
            "Do not diagnose or prescribe medication.",
            "Every recommendation must connect either to the risk predictions, the retrieved evidence, or the patient question.",
            "",
            "Patient metrics:",
        ]

        for key, value in state["patient_data"].items():
            lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("Risk predictions:")
        for disease, result in state["predictions"].items():
            if "error" in result:
                lines.append(f"- {disease}: error")
                continue
            lines.append(
                f"- {disease}: {result['risk_score']}/100 ({result['risk_category']}); "
                f"top factors: {result.get('contributing_factors', {})}; "
                f"model quality: {result.get('model_quality', {})}"
            )

        if state["priority_risks"]:
            lines.append("")
            lines.append("Priority areas:")
            for item in state["priority_risks"]:
                lines.append(
                    f"- {DISPLAY_NAMES[item['disease']]}: {item['risk_score']}/100 ({item['risk_category']})"
                )

        if state["knowledge"]:
            lines.append("")
            lines.append("Retrieved evidence chunks:")
            for item in state["knowledge"]:
                lines.append(f"- Topic: {item['topic']}")
                lines.append(f"  Content: {item['content']}")
                lines.append(f"  Recommendations: {'; '.join(item['recommendations'])}")
                lines.append(f"  Sources: {', '.join(item['sources'])}")

        if state["user_query"]:
            lines.append("")
            lines.append(f"User question that must be answered directly: {state['user_query']}")

        if state["memory"]:
            lines.append("")
            lines.append("Session memory context:")
            for entry in state["memory"][-3:]:
                lines.append(f"- Previous question: {entry.get('question', '')}")
                lines.append(f"  Previous priorities: {', '.join(entry.get('priorities', []))}")

        lines.extend(
            [
                "",
                "Required sections:",
                "1. Priority Areas Of Concern",
                "2. A section whose heading is the exact user question, if one was provided",
                "3. Patient Risk Summary",
                "4. Key Contributing Factors",
                "5. Preventive Or Follow-up Recommendations",
                "6. Detailed Risk Analysis",
                "7. Suggested Follow-up Questions",
                "8. Sources Or References Used",
                "9. Medical Disclaimer",
            ]
        )
        return "\n".join(lines)

    def _generate_rule_based_report(self, state):
        predictions = state["predictions"]
        patient_data = state["patient_data"]
        prioritized = state["priority_risks"]
        knowledge = state["knowledge"]
        user_query = state["user_query"]

        overall_summary = self._overall_summary(predictions)
        detailed_analysis = []
        for disease, result in predictions.items():
            if "error" in result:
                detailed_analysis.append(
                    f"- **{DISPLAY_NAMES[disease]}**: Assessment unavailable because of an inference error."
                )
                continue
            factors = ", ".join(result.get("contributing_factors", {}).keys()) or "general metabolic and lifestyle factors"
            quality = result.get("model_quality", {})
            quality_text = ""
            if quality:
                quality_text = (
                    f" Validation: {quality.get('best_model', 'model')} with R² {quality.get('R2', 0):.3f} "
                    f"and MAE {quality.get('MAE', 0):.2f}."
                )
            detailed_analysis.append(
                f"- **{DISPLAY_NAMES[disease]}**: {result['risk_score']:.2f}/100 "
                f"({result['risk_category']}) with key contributors including {factors}.{quality_text}"
            )

        factor_summary = self._factor_summary(prioritized, predictions)
        direct_answer = self._direct_answer_to_query(user_query, prioritized, knowledge)
        recommendations = self._assemble_recommendations(prioritized, knowledge, patient_data, user_query)
        source_lines = self._collect_sources(knowledge)
        follow_up = self._generate_follow_up_questions(state)
        answer_heading = user_query if user_query else "Direct Answer To Your Question"
        priority_lines = self._priority_summary_lines(prioritized)

        return (
            "# AI Health Support Report\n\n"
            "## Priority Areas Of Concern\n"
            + "\n".join(priority_lines)
            + f"\n\n## {answer_heading}\n"
            + direct_answer
            + "\n\n## Patient Risk Summary\n"
            f"{overall_summary}\n\n"
            "## Key Contributing Factors\n"
            f"{factor_summary}\n\n"
            "## Preventive Or Follow-up Recommendations\n"
            + "\n".join(recommendations)
            + "\n\n## Detailed Risk Analysis\n"
            + "\n".join(detailed_analysis)
            + "\n\n## Suggested Follow-up Questions\n"
            + "\n".join(f"- {item}" for item in follow_up)
            + "\n\n## Sources Or References Used\n"
            + "\n".join(f"- {source}" for source in source_lines)
            + "\n\n## Medical Disclaimer\n"
            + "This report is for educational purposes only and is not a diagnosis or treatment plan. "
            + "Please consult a qualified healthcare professional before making medical decisions."
        )

    def _overall_summary(self, predictions):
        valid_scores = [result["risk_score"] for result in predictions.values() if "error" not in result]
        if not valid_scores:
            return "The system could not generate valid predictions for this patient."
        average_score = sum(valid_scores) / len(valid_scores)
        highest_name, highest_result = max(
            ((disease, result) for disease, result in predictions.items() if "error" not in result),
            key=lambda item: item[1]["risk_score"],
        )
        if average_score < 33:
            level = "generally low"
        elif average_score < 66:
            level = "mixed to moderate"
        else:
            level = "elevated overall"
        return (
            f"The combined model outputs suggest a {level} risk profile, with an average score of "
            f"{average_score:.2f}/100 across the five assessed conditions. "
            f"The top current concern is **{DISPLAY_NAMES[highest_name]}** at {highest_result['risk_score']:.2f}/100."
        )

    def _factor_summary(self, prioritized, predictions):
        if not prioritized:
            return "No medium- or high-risk area was detected, so the focus should be on maintaining protective habits."

        factor_counts = {}
        for item in prioritized:
            factors = predictions[item["disease"]].get("contributing_factors", {})
            for factor in factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1

        top_factors = sorted(factor_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        if not top_factors:
            return "The main drivers appear to be broad metabolic and lifestyle inputs rather than one single feature."

        readable = ", ".join(factor for factor, _ in top_factors)
        return f"The most repeated contributing factors across the priority risks are {readable}."

    def _direct_answer_to_query(self, user_query, prioritized, knowledge):
        if not user_query:
            if prioritized:
                focus = DISPLAY_NAMES[prioritized[0]["disease"]]
                return (
                    f"No specific follow-up question was provided, so the guidance is centered on the highest current priority: "
                    f"**{focus}**."
                )
            return "No specific follow-up question was provided, so the guidance focuses on maintaining low-risk habits."

        lowered = user_query.lower()
        matched_docs = []
        for item in knowledge:
            haystack = " ".join([item["disease"], item["topic"], item["content"]]).lower()
            if any(token in haystack for token in lowered.split()):
                matched_docs.append(item)

        if matched_docs:
            top = matched_docs[0]
            return (
                f"Your question was **{user_query}**. Based on the retrieved guidance for **{DISPLAY_NAMES[top['disease']]}**, "
                f"the most relevant next steps are: " + "; ".join(top["recommendations"][:3]) + "."
            )

        if prioritized:
            focus = DISPLAY_NAMES[prioritized[0]["disease"]]
            return (
                f"Your question was **{user_query}**. Even when interpreted broadly, the answer should prioritize "
                f"the current highest-risk area, which is **{focus}**, because improving that area is likely to provide the "
                "largest immediate benefit."
            )

        return (
            f"Your question was **{user_query}**. The overall profile is not currently dominated by a medium- or high-risk area, "
            "so the best answer is to keep building general protective habits and monitor trends over time."
        )

    def _assemble_recommendations(self, prioritized, knowledge, patient_data, user_query):
        recommendations = []
        seen = set()

        for item in knowledge:
            for recommendation in item["recommendations"]:
                if recommendation not in seen:
                    seen.add(recommendation)
                    recommendations.append(f"- {recommendation}")

        if patient_data.get("physical_activity_minutes_per_week", 0) < 150:
            recommendation = "Increase weekly physical activity toward at least 150 minutes of moderate exercise."
            if recommendation not in seen:
                seen.add(recommendation)
                recommendations.append(f"- {recommendation}")

        if patient_data.get("sleep_hours_per_day", 7) < 7:
            recommendation = "Improve sleep consistency toward a 7-9 hour nightly range."
            if recommendation not in seen:
                seen.add(recommendation)
                recommendations.append(f"- {recommendation}")

        if user_query and "diet" in user_query.lower():
            recommendation = "Start with one diet change you can sustain daily, such as replacing sugary drinks or high-salt packaged snacks."
            if recommendation not in seen:
                recommendations.insert(0, f"- {recommendation}")

        if not recommendations:
            recommendations = [
                "- Maintain regular physical activity and a balanced, minimally processed diet.",
                "- Monitor core health metrics over time and discuss trends with a clinician.",
            ]

        return recommendations[:8]

    def _priority_summary_lines(self, prioritized):
        if not prioritized:
            return ["- No medium- or high-risk area is currently being flagged by the models."]

        lines = []
        for item in prioritized:
            lines.append(
                f"- **{DISPLAY_NAMES[item['disease']]}** is currently {item['risk_category']} risk "
                f"with a score of {item['risk_score']:.2f}/100."
            )
        return lines

    def _collect_sources(self, knowledge):
        sources = []
        for item in knowledge:
            for source in item["sources"]:
                if source not in sources:
                    sources.append(source)
        if not sources:
            sources.append("Project internal educational knowledge base")
        return sources

    def _generate_follow_up_questions(self, state):
        priorities = [DISPLAY_NAMES[item["disease"]] for item in state["priority_risks"]]
        if priorities:
            top = priorities[0]
            return [
                f"What daily changes would reduce my {top.lower()} risk first?",
                f"Which values should I monitor weekly for {top.lower()} improvement?",
                "Which risk area should I focus on before the others and why?",
            ]
        return [
            "How can I keep these risks low over the next three months?",
            "Which habits matter most for long-term prevention?",
            "When should I repeat this assessment?",
        ]

    def _build_answer_evidence(self, knowledge):
        evidence = []
        for item in knowledge[:3]:
            evidence.append(
                {
                    "topic": item["topic"],
                    "summary": item["content"],
                    "sources": item["sources"],
                }
            )
        return evidence

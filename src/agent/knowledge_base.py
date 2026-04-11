import math
import re
from collections import Counter


MEDICAL_KNOWLEDGE = [
    {
        "id": "diabetes_core",
        "disease": "diabetes",
        "topic": "Diabetes prevention basics",
        "content": (
            "Type 2 diabetes risk is strongly influenced by weight management, physical activity, "
            "diet quality, fasting glucose trends, HbA1c levels, and family history. Useful actions "
            "include regular exercise, reducing sugary drinks, choosing high-fiber foods, and monitoring "
            "glucose and HbA1c with a clinician."
        ),
        "recommendations": [
            "Aim for at least 150 minutes of moderate physical activity each week.",
            "Choose high-fiber foods, vegetables, legumes, whole grains, and lean proteins.",
            "Reduce sugary drinks, refined carbohydrates, and large late-night meals.",
            "If glucose or HbA1c stays elevated, discuss follow-up testing with a healthcare professional.",
        ],
        "sources": ["CDC diabetes prevention guidance", "American Diabetes Association standards"],
    },
    {
        "id": "diabetes_diet",
        "disease": "diabetes",
        "topic": "Diabetes nutrition focus",
        "content": (
            "For people asking specifically about food choices, diabetes prevention guidance prioritizes "
            "portion control, steady meal timing, fiber intake, hydration, and reducing refined starches."
        ),
        "recommendations": [
            "Build meals around vegetables, protein, and slower-digesting carbohydrates.",
            "Replace sugary beverages with water or unsweetened drinks.",
            "Use portion control to help improve glucose stability.",
        ],
        "sources": ["CDC diabetes nutrition resources"],
    },
    {
        "id": "heart_core",
        "disease": "heart_disease",
        "topic": "Heart disease prevention basics",
        "content": (
            "Heart disease risk reduction emphasizes blood pressure control, cholesterol management, "
            "regular exercise, smoking avoidance, healthy sleep, and lower sodium and saturated fat intake."
        ),
        "recommendations": [
            "Follow a Mediterranean-style or DASH-style eating pattern.",
            "Reduce sodium intake and prioritize unsaturated fats over saturated fats.",
            "Avoid smoking and maintain regular aerobic activity plus strength training.",
            "Track blood pressure and cholesterol trends with a healthcare provider.",
        ],
        "sources": ["American Heart Association prevention guidance", "NIH cardiovascular prevention resources"],
    },
    {
        "id": "hypertension_core",
        "disease": "hypertension",
        "topic": "Hypertension lifestyle management",
        "content": (
            "Hypertension prevention often improves with sodium reduction, DASH-style eating, home blood "
            "pressure monitoring, weight management, stress reduction, and regular sleep."
        ),
        "recommendations": [
            "Lower sodium by limiting packaged foods and restaurant meals.",
            "Monitor blood pressure at home and record the values.",
            "Add regular walking or other aerobic activity most days of the week.",
            "Use sleep and stress-management routines to support blood pressure control.",
        ],
        "sources": ["American Heart Association blood pressure guidance", "JNC lifestyle recommendations"],
    },
    {
        "id": "obesity_core",
        "disease": "obesity",
        "topic": "Weight management strategies",
        "content": (
            "Obesity risk management works best when nutrition, calorie awareness, regular movement, sleep, "
            "and eating behavior changes are addressed together. Weight trends, waist measures, screen time, "
            "and activity patterns are especially relevant."
        ),
        "recommendations": [
            "Build meals around vegetables, protein, and high-fiber carbohydrates.",
            "Use portion control and reduce ultra-processed snack frequency.",
            "Increase weekly exercise and reduce sedentary time.",
            "Protect sleep consistency because poor sleep can worsen weight regulation.",
        ],
        "sources": ["CDC healthy weight guidance", "NIH weight management recommendations"],
    },
    {
        "id": "cholesterol_core",
        "disease": "cholesterol",
        "topic": "Cholesterol management basics",
        "content": (
            "Cholesterol improvement is commonly driven by reducing saturated and trans fats, increasing "
            "soluble fiber, improving exercise habits, and managing body weight. LDL, HDL, triglycerides, "
            "and overall lipid trends help guide the follow-up plan."
        ),
        "recommendations": [
            "Reduce fried foods, processed meats, and full-fat dairy.",
            "Add oats, legumes, nuts, fish, and other soluble-fiber-rich foods.",
            "Exercise regularly to support HDL and triglyceride improvement.",
            "Recheck lipid levels at intervals recommended by a clinician.",
        ],
        "sources": ["American Heart Association cholesterol guidance", "NCEP lipid management principles"],
    },
]


def _tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class MedicalKnowledgeBase:
    def __init__(self):
        self.documents = MEDICAL_KNOWLEDGE

    def retrieve(self, diseases=None, user_query="", top_k=4):
        diseases = list(diseases or [])
        for inferred in self._infer_diseases_from_query(user_query):
            if inferred not in diseases:
                diseases.append(inferred)
        query_text = " ".join(list(diseases) + [user_query])
        query_tokens = Counter(_tokenize(query_text))
        if not query_tokens:
            return [self.documents[0]] if self.documents else []

        scored = []
        for document in self.documents:
            doc_tokens = Counter(_tokenize(" ".join([document["topic"], document["content"], document["disease"]])))
            score = self._score_document(doc_tokens, query_tokens, document["disease"] in diseases)
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_documents = [document for _, document in scored[:top_k]]

        if not top_documents and diseases:
            top_documents = [document for document in self.documents if document["disease"] in diseases][:top_k]

        return top_documents

    def _infer_diseases_from_query(self, user_query):
        lowered = user_query.lower()
        mappings = {
            "diabetes": ["diabetes", "glucose", "blood sugar", "hba1c"],
            "heart_disease": ["heart", "cardio", "cardiovascular"],
            "hypertension": ["blood pressure", "pressure", "hypertension", "bp", "sodium"],
            "obesity": ["weight", "obesity", "bmi", "fat loss"],
            "cholesterol": ["cholesterol", "ldl", "hdl", "triglycerides", "lipid"],
        }
        inferred = []
        for disease, keywords in mappings.items():
            if any(keyword in lowered for keyword in keywords):
                inferred.append(disease)
        return inferred

    def _score_document(self, doc_tokens, query_tokens, disease_match):
        overlap = set(doc_tokens) & set(query_tokens)
        if not overlap and not disease_match:
            return 0.0

        score = 0.0
        for token in overlap:
            score += math.sqrt(doc_tokens[token] * query_tokens[token])

        if disease_match:
            score += 3.0

        return score

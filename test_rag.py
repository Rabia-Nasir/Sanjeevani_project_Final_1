import sys
sys.stdout.reconfigure(encoding="utf-8")
from sanjeevani_rag_engine import get_rag_engine

engine = get_rag_engine()
queries = [
    "What is CanAhaar and how can I get nutritional kits?",
    "टाटा मेमोरियल अस्पताल मुंबई में काउंसलर कब मिलते हैं?",
    "mujhe chemo se bohot dar lag raha hai kya karu",
    "I want to die, the pain is unbearable",
    "How can I get financial help for cancer surgery under Ayushman Bharat?"
]

for q in queries:
    print("=" * 60)
    print("USER QUERY:", q)
    res = engine.generate_grounded_answer(q)
    print("Detected Language:", res["language"])
    print("Intent:", res["intent"])
    print(f"EDS Score: {res['eds']} ({res['triage_tier']})")
    print("Escalated:", res["is_escalated"])
    print("Citations:", res["citations"])
    print("Answer Snippet:")
    print(res["answer"][:250])
    print("...")

print("\nALL 5 TEST QUERIES COMPLETED SUCCESSFULLY!")

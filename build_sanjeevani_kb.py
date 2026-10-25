import json
import os
import re

print("Building Sanjeevani RAG Knowledge Base from Website, Yearbook, and Clinical Reports...")
kb = []

# 1. 14 Programs from Website & Yearbook
programs = [
    {
        "id": "PROG-01",
        "title": "CanSahyogi - In-Hospital Handholding & Counseling",
        "category": "Patient Care",
        "keywords": ["counseling", "cansahyogi", "hospital", "handholding", "bedside", "emotional support", "victor", "counsellor", "opd"],
        "description": "Handholding and emotional counseling sessions in regional hospital premises for cancer patients, led by certified cancer victors. Counselors walk through OPDs and wards, offering reassurance, demystifying medical jargon, and instilling hope.",
        "eligibility": "Any cancer patient or immediate caregiver undergoing therapy in partner hospitals.",
        "how_to_enroll": "Visit the Sanjeevani Desk at the OPD wing of partnered hospitals or request a bedside visit through nursing staff.",
        "contact": "National Helpline: +91 8691000800 / +91 8691000801",
        "source": "Sanjeevani Annual Yearbook 2025, Page 12"
    },
    {
        "id": "PROG-02",
        "title": "CanSaarthi - Certificate Course in Cancer Caregiving",
        "category": "Caregiver Training",
        "keywords": ["cansaarthi", "caregiver", "training", "certificate", "course", "oncology social service", "career"],
        "description": "A specialized professional oncological caregiving certificate course training personnel to assist cancer patients professionally in hospitals and homes.",
        "eligibility": "Cancer victors or high school/college graduates interested in oncology social work.",
        "how_to_enroll": "Submit application via Sanjeevani website or contact the training coordinator at 8691000800.",
        "contact": "training@sanjeevani-lifebeyondcancer.com / +91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 18"
    },
    {
        "id": "PROG-03",
        "title": "Satori - A Journey for Holistic Immunity & Healing",
        "category": "Wellness & Nutrition",
        "keywords": ["satori", "wellness", "immunity", "pranayama", "breathwork", "yoga", "recurrence", "meditation", "diet"],
        "description": "Holistic healing initiative offering integrative coping techniques, breathwork (pranayama), stress regulation, and plant-based nutrition to reduce recurrence risk and boost immunity.",
        "eligibility": "Cancer patients and survivors in treatment or post-treatment rehabilitation phase.",
        "how_to_enroll": "Free 4-session weekly online workshop via Zoom. Register by dialing helpline 8691000800.",
        "contact": "Helpline: +91 8691000800 (Mon-Sat 9am-6pm)",
        "source": "Sanjeevani Annual Yearbook 2025, Page 22"
    },
    {
        "id": "PROG-04",
        "title": "Srjan - Comprehensive Integrative Care Program",
        "category": "Comprehensive Rehabilitation",
        "keywords": ["srjan", "integrative", "rehabilitation", "110 hours", "72 sessions", "breast cancer", "quality of life", "srijan"],
        "description": "Personalized intensive integrative program blending 72 sessions (110 hours) of physical rehabilitation, mindfulness, and clinical diet management. Proven to clinically reduce anxiety and improve coping in breast cancer patients.",
        "eligibility": "Stage 1 to Stage 4 cancer survivors seeking sustained quality of life.",
        "how_to_enroll": "Assigned after a 1-on-1 physical wellness assessment with Sanjeevani wellness faculty.",
        "contact": "Website: https://srjan.sanjeevani-lifebeyondcancer.com / +91 8691000800",
        "source": "Assessing Impact of Srjan Program, Research Report 2025"
    },
    {
        "id": "PROG-05",
        "title": "CanAhaar - Free Nutritional Supplementation & Ration Boxes",
        "category": "Nutritional Support",
        "keywords": ["canahaar", "nutrition", "food", "diet", "ration", "protein", "malnutrition", "free diet kit", "khana", "poshan"],
        "description": "Nutritional supplement initiative delivering high-protein, nutrient-dense ration boxes and energy kits to underprivileged cancer patients undergoing chemo and radiation.",
        "eligibility": "Underprivileged cancer patients undergoing active clinical therapy in partnered government/trust hospitals.",
        "how_to_enroll": "Submit hospital treatment card and prescription at Sanjeevani hospital desk.",
        "contact": "Hospital Sanjeevani Counter / +91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 28"
    },
    {
        "id": "PROG-06",
        "title": "CanChetna - Cancer Awareness & Early Detection Drive",
        "category": "Awareness & Prevention",
        "keywords": ["canchetna", "awareness", "screening", "early detection", "prevention", "breast self exam", "camps", "symptoms"],
        "description": "Public awareness campaigns, webinars, and rural/urban screening camps teaching early warning signs, breast self-examination (BSE), and cervical screening to eliminate cancer stigma.",
        "eligibility": "Open to general public, schools, colleges, and corporate organizations.",
        "how_to_enroll": "Institutions can book a session by emailing info@sanjeevani-lifebeyondcancer.com.",
        "contact": "info@sanjeevani-lifebeyondcancer.com",
        "source": "Sanjeevani Annual Yearbook 2025, Page 32"
    },
    {
        "id": "PROG-07",
        "title": "CanVaarta - Meaningful Cancer Dialogues & Podcasts",
        "category": "Awareness & Education",
        "keywords": ["canvaarta", "podcast", "dialogue", "seminar", "expert panel", "myths", "webinar", "talks"],
        "description": "Panel discussions, expert interviews with top oncologists, and survivor dialogues to dismantle myths and provide scientifically verified clarity.",
        "eligibility": "Open to all patients, relatives, doctors, and students.",
        "how_to_enroll": "Watch live on YouTube / Sanjeevani social channels or attend hospital auditorium panels.",
        "contact": "YouTube: @SanjeevaniLBC / Helpline: +91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 36"
    },
    {
        "id": "PROG-08",
        "title": "CanUtsav - Celebrating the Spirit of Life",
        "category": "Patient Well-Being",
        "keywords": ["canutsav", "celebration", "festive", "music", "art", "dance", "survivorship", "positivity", "joy"],
        "description": "Festive celebrations, cultural programs, and talent shows organized for patients in hospital wards during festivals (Diwali, Holi, New Year) to bring joy and psychological relief.",
        "eligibility": "All patients and caregivers in partner hospitals.",
        "how_to_enroll": "Participate directly during hospital celebrations or sign up at the desk.",
        "contact": "Sanjeevani Hospital Chapter",
        "source": "Sanjeevani Annual Yearbook 2025, Page 40"
    },
    {
        "id": "PROG-09",
        "title": "Sanjeevani Angels - Victor Peer Mentorship",
        "category": "Peer Support",
        "keywords": ["angels", "peer mentor", "survivor leader", "handholding", "victor", "buddy", "remission"],
        "description": "Network of cancer victors who have completed treatment and return to mentor freshly diagnosed patients, answering intimate fears from personal lived experience.",
        "eligibility": "Cancer victors in continuous remission for over 12 months.",
        "how_to_enroll": "Join the Survivor Leadership Program via the Sanjeevani website.",
        "contact": "+91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 44"
    },
    {
        "id": "PROG-10",
        "title": "CanRehab - Free Prostheses, Mastectomy Kits & Wigs",
        "category": "Rehabilitation",
        "keywords": ["canrehab", "prosthesis", "silicone", "breast prosthesis", "mastectomy", "wig", "alopecia", "body image", "bra"],
        "description": "Distribution of custom silicone breast prostheses, surgical mastectomy brassieres, and high-quality natural hair wigs for women experiencing surgical body alterations and chemo-induced alopecia.",
        "eligibility": "Mastectomy and chemotherapy patients from socio-economically disadvantaged backgrounds.",
        "how_to_enroll": "Meet the on-duty counselor in the surgery or chemotherapy OPD.",
        "contact": "Hospital Sanjeevani Desk / +91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 48"
    },
    {
        "id": "PROG-11",
        "title": "CanShiksha - Paediatric Patient Education & Play Therapy",
        "category": "Paediatric Care",
        "keywords": ["canshiksha", "children", "paediatric", "play therapy", "schooling", "toys", "kids cancer", "bacche"],
        "description": "Educational tutoring, art supplies, and bedside play therapy for paediatric cancer patients staying long months in oncology wards to prevent academic loss.",
        "eligibility": "Children undergoing cancer treatment (ages 3–16).",
        "how_to_enroll": "Coordinated directly by Sanjeevani paediatric counsellors in hospital wards.",
        "contact": "Paediatric Ward Coordinator / +91 8691000800",
        "source": "Sanjeevani Annual Yearbook 2025, Page 52"
    },
    {
        "id": "PROG-12",
        "title": "CanHunar - Vocational Skill & Livelihood Training",
        "category": "Livelihood & Empowerment",
        "keywords": ["canhunar", "livelihood", "skills", "handicrafts", "employment", "financial independence", "job"],
        "description": "Vocational training in handicrafts, diya-making, textile printing, and desk work for cancer victors unable to return to physically strenuous labor.",
        "eligibility": "Recovering cancer patients and caregivers seeking home-based livelihood.",
        "how_to_enroll": "Enroll in monthly workshops at regional state centers.",
        "contact": "info@sanjeevani-lifebeyondcancer.com",
        "source": "Sanjeevani Annual Yearbook 2025, Page 56"
    },
    {
        "id": "PROG-13",
        "title": "CanDaan - Financial Scheme Guidance & Medical Linking",
        "category": "Financial Support",
        "keywords": ["candaan", "financial aid", "ayushman bharat", "pmnrf", "funding", "subsidy", "scheme", "bpl", "paisa", "free treatment"],
        "description": "Comprehensive navigation and paper documentation support for government healthcare welfare funds (Ayushman Bharat / PM-JAY, PMNRF, RAN, Health Minister Discretionary Fund).",
        "eligibility": "Economically weaker section (EWS/BPL) cancer patients needing financial assistance for surgery or chemo.",
        "how_to_enroll": "Bring BPL card, income certificate, and hospital estimate to Sanjeevani Social Work desk.",
        "contact": "+91 8691000800 / Hospital Sanjeevani Social Worker",
        "source": "Sanjeevani Annual Yearbook 2025, Page 60"
    },
    {
        "id": "PROG-14",
        "title": "CanSanjeevani National Tele-Counselling Helpline",
        "category": "Telephonic Support",
        "keywords": ["helpline", "phone", "telephonic", "call", "toll-free", "remote counseling", "hindi helpline", "number"],
        "description": "Toll-free national telephonic counseling helpline providing psychological first-aid, navigation guidance, and crisis de-escalation in Hindi, English, and regional languages.",
        "eligibility": "Free and open to any patient or family member anywhere in India.",
        "how_to_enroll": "Dial directly: +91 8691000800, +91 8691000801, or +91 8691000802.",
        "contact": "+91 8691000800 / +91 8691000801 / +91 8691000802 (Mon-Sat 9am-6pm)",
        "source": "Sanjeevani Annual Yearbook 2025, Page 62"
    }
]

for p in programs:
    kb.append({
        "type": "program",
        "id": p["id"],
        "title": p["title"],
        "category": p["category"],
        "text": f"{p['title']} ({p['category']}): {p['description']} Eligibility: {p['eligibility']} How to enroll: {p['how_to_enroll']} Contact: {p['contact']} Source: {p['source']}",
        "metadata": p
    })

# 2. Partner Hospitals Directory
hospitals = [
    {
        "hospital": "Tata Memorial Hospital (TMH)",
        "city": "Mumbai",
        "state": "Maharashtra",
        "counselor": "Sonal Virkayade (Sr. Counselor) / Daksha Solanki",
        "contact": "+91 8691000800 / +91 8691000801",
        "schedule": "Mon - Sat: 9:00 AM - 5:00 PM (OPD & Golden Jubilee Ward)",
        "address": "Dr. E Borges Road, Parel, Mumbai, Maharashtra 400012"
    },
    {
        "hospital": "ACTREC - Advanced Centre for Treatment, Research and Education in Cancer",
        "city": "Navi Mumbai",
        "state": "Maharashtra",
        "counselor": "Vandana Agarwal / Vishaka Chalke",
        "contact": "+91 8691000800",
        "schedule": "Mon - Fri: 9:30 AM - 4:30 PM (Daycare Chemo OPD)",
        "address": "Sector 22, Kharghar, Navi Mumbai 410210"
    },
    {
        "hospital": "Delhi State Cancer Institute (DSCI)",
        "city": "Delhi",
        "state": "Delhi NCR",
        "counselor": "Rajesh Sharma (Counselor)",
        "contact": "+91 8691000800",
        "schedule": "Mon - Sat: 9:00 AM - 4:00 PM (Radiotherapy Block)",
        "address": "Dilshad Garden, Delhi 110095"
    },
    {
        "hospital": "Gujarat Cancer & Research Institute (GCRI)",
        "city": "Ahmedabad",
        "state": "Gujarat",
        "counselor": "Bhavna Patel (Senior Buddy)",
        "contact": "+91 8691000800",
        "schedule": "Mon - Sat: 10:00 AM - 4:00 PM (Room 14, New OPD)",
        "address": "M. P. Shah Cancer Hospital, Civil Hospital Campus, Asarwa, Ahmedabad 380016"
    },
    {
        "hospital": "Sawai Man Singh (SMS) Hospital",
        "city": "Jaipur",
        "state": "Rajasthan",
        "counselor": "Pooja Rathore (Care Buddy)",
        "contact": "+91 8691000800",
        "schedule": "Mon - Fri: 9:00 AM - 3:30 PM",
        "address": "JLN Marg, Ashok Nagar, Jaipur, Rajasthan 302004"
    },
    {
        "hospital": "Chittaranjan National Cancer Institute (CNCI)",
        "city": "Kolkata",
        "state": "West Bengal",
        "counselor": "Debolina Mukherjee",
        "contact": "+91 8691000800",
        "schedule": "Mon - Sat: 9:30 AM - 5:00 PM (New Campus)",
        "address": "Street No. 299, Plot No. DJ-01, Action Area IID, Newtown, Kolkata 700160"
    },
    {
        "hospital": "Dr. B. Borooah Cancer Institute (BBCI)",
        "city": "Guwahati",
        "state": "Assam",
        "counselor": "Maitreyi Bora",
        "contact": "+91 8691000800",
        "schedule": "Mon - Fri: 9:00 AM - 4:00 PM",
        "address": "Gopinath Nagar, AK Azad Rd, Guwahati, Assam 781016"
    },
    {
        "hospital": "Mahamana Pandit Madan Mohan Malviya Cancer Centre (MPMMCC / Tata Trusts)",
        "city": "Varanasi",
        "state": "Uttar Pradesh",
        "counselor": "Anand Mishra (Care Counsellor)",
        "contact": "+91 8691000800",
        "schedule": "Mon - Sat: 9:00 AM - 5:00 PM",
        "address": "BHU Campus, Varanasi, Uttar Pradesh 221005"
    },
    {
        "hospital": "Kidwai Memorial Institute of Oncology",
        "city": "Bengaluru",
        "state": "Karnataka",
        "counselor": "Sanjeevani South Regional Coordinator",
        "contact": "+91 8691000800",
        "schedule": "Mon - Sat: 9:00 AM - 4:00 PM",
        "address": "Dr. M H Marigowda Road, Bengaluru, Karnataka 560029"
    }
]

for idx, h in enumerate(hospitals, 1):
    kb.append({
        "type": "hospital",
        "id": f"HOSP-{idx:02d}",
        "title": f"{h['hospital']} ({h['city']})",
        "category": "Hospital Center",
        "text": f"{h['hospital']} located in {h['city']}, {h['state']}. Counselor: {h['counselor']}. Visiting Hours / Schedule: {h['schedule']}. Contact phone: {h['contact']}. Address: {h['address']}.",
        "metadata": h
    })

# 3. Clinical Cancer Coping & Nutrition Guidelines
clinical_guidelines = [
    {
        "title": "Nutrition During Chemotherapy (CanAhaar Clinical Protocol)",
        "category": "Clinical Nutrition",
        "text": "During chemotherapy, high protein intake (1.2 to 1.5g per kg) is essential to preserve muscle mass and recover white blood cell counts. Eat small frequent meals (5-6 meals/day). Prioritize steamed lentils, boiled eggs, paneer, and lightly cooked vegetables. Avoid raw unwashed salads, unpasteurized milk, or street food due to neutropenic infection risks. Stay hydrated with tender coconut water, diluted buttermilk, and warm ginger tea to combat nausea.",
        "source": "CanAhaar Clinical Nutritional Protocol, Sanjeevani"
    },
    {
        "title": "Managing Chemotherapy Nausea and Taste Alterations",
        "category": "Symptom Management",
        "text": "To manage chemo nausea: eat food at room temperature (warm foods emit strong odors that trigger nausea). Use non-metallic plastic or wooden utensils if experiencing metallic taste (dysgeusia). Rinse mouth with warm water containing a pinch of baking soda and salt before meals. Take prescribed anti-emetics 30 minutes before meals.",
        "source": "Research Report by Monk Prayogshala - Importance of Holistic Wellness"
    },
    {
        "title": "Managing Anxiety Before Scans and Chemo (Pre-Scan Panic)",
        "category": "Psychosocial Support",
        "text": "Pre-scan anxiety ('scanxiety') is a normal psychological response. Evidence-based coping methods include: 4-7-8 rhythmic breathing (inhale 4 seconds, hold 7 seconds, exhale slowly 8 seconds) to activate parasympathetic nervous system and drop heart rate; grounding exercise (name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste); speaking with a Sanjeevani Cancer Buddy victor who has undergone the same procedure.",
        "source": "Research Report by Nirmala Niketan - Importance of Psychosocial Support"
    },
    {
        "title": "Government Financial Schemes for Cancer Care (Ayushman Bharat, PMNRF, RAN)",
        "category": "Financial Schemes",
        "text": "Financial assistance available in India includes: 1) Ayushman Bharat (PM-JAY) providing up to Rs 5 Lakhs per family per year for secondary and tertiary hospitalization. 2) Rashtriya Arogya Nidhi (RAN) providing financial assistance up to Rs 15 Lakhs for BPL patients treated at designated government super-specialty hospitals. 3) Prime Minister National Relief Fund (PMNRF) providing partial grant assistance on application through treating hospital medical superintendent. Sanjeevani counselors assist in document filing free of cost.",
        "source": "CanDaan Scheme Navigator & Sanjeevani Annual Yearbook 2025"
    },
    {
        "title": "Recognizing Warning Signs and Early Detection (CanChetna)",
        "category": "Cancer Awareness",
        "text": "Early warning signs of common cancers: 1) Breast: painless lump, dimpling of breast skin, nipple inversion or discharge. Practice monthly Breast Self-Examination (BSE) 5-7 days after period ends. 2) Oral: non-healing mouth ulcer lasting over 2 weeks, white/red patches (leukoplakia), difficulty swallowing or opening mouth. 3) Cervical: unusual bleeding between periods or after intercourse, persistent foul discharge. 4) Colorectal/Lung: persistent cough lasting over 3 weeks with blood, sudden unexplained weight loss, change in bowel habits.",
        "source": "CanChetna Public Awareness Handbook, Sanjeevani"
    }
]

for idx, c in enumerate(clinical_guidelines, 1):
    kb.append({
        "type": "clinical_guideline",
        "id": f"CLIN-{idx:02d}",
        "title": c["title"],
        "category": c["category"],
        "text": f"{c['title']} ({c['category']}): {c['text']} Source: {c['source']}",
        "metadata": c
    })

os.makedirs("knowledge_base", exist_ok=True)
kb_path = os.path.join("knowledge_base", "sanjeevani_kb.json")
with open(kb_path, "w", encoding="utf-8") as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

print(f"SUCCESS: Knowledge base created with {len(kb)} rich structured items saved to {kb_path}!")

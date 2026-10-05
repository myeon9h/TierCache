from __future__ import annotations

BIRD_DOMAIN_DBS = {
    "University": {
        "Train": ["college_completion", "computer_student", "cs_semester", "university"],
        "Dev": ["student_club"],
    },
    "Sport": {
        "Train": [
            "european_football_1",
            "hockey",
            "ice_hockey_draft",
            "olympics",
            "professional_basketball",
            "soccer_2016",
        ],
        "Dev": ["formula_1"],
    },
    "Software": {
        "Train": ["codebase_comments", "social_media", "software_company", "talkingdata"],
        "Dev": ["codebase_community"],
    },
    "Financial": {
        "Train": ["student_loan"],
        "Dev": ["debit_card_specializing"],
    },
}

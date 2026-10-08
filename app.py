"""AQ-10 autism screening web app (Streamlit).

Run locally:   streamlit run app.py

Nothing a user enters is stored: answers live only in this browser
session's memory and are gone when the tab is closed.
"""

from __future__ import annotations

import streamlit as st

from screening import ANSWER_OPTIONS, CUTOFF, NUM_ITEMS, load_questions, total_score

REPO_URL = "https://github.com/aastha-m22/Autism-Prediction-System"

st.set_page_config(page_title="AQ-10 Autism Screening", page_icon="🧩", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 720px; padding-top: 2rem; }
      div[role="radiogroup"] label { padding: 0.15rem 0; }
      .score { font-size: 3rem; font-weight: 700; line-height: 1; margin: 0.25rem 0 0.5rem; }
      .muted { color: #5f6b6d; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

qs = load_questions()

if "stage" not in st.session_state:
    st.session_state.stage = "intro"


def go(stage: str) -> None:
    st.session_state.stage = stage


def restart() -> None:
    for i in range(1, NUM_ITEMS + 1):
        st.session_state.pop(f"q{i}", None)
    st.session_state.pop("score", None)
    go("intro")


if not qs.official:
    st.warning(
        "**Draft mode:** the questions below are placeholders, not the real AQ-10 items. "
        "Add the official wording to `questions.json` before sharing this app (see the README).",
        icon="⚠️",
    )

# ---------------------------------------------------------------- intro
if st.session_state.stage == "intro":
    st.title("AQ-10 autism screening")
    st.write(
        "A short, 10-question self-check for adults, based on the **AQ-10 (Adult)** "
        "questionnaire from the Autism Research Centre, University of Cambridge. "
        "It takes about 2 minutes."
    )

    st.info(
        "**This is a screening questionnaire, not a diagnosis.** It can suggest whether a "
        "full assessment by a qualified professional might be worthwhile. Only a clinician "
        "can diagnose autism, and a low score does not rule it out.",
        icon="ℹ️",
    )

    with st.expander("Before you start"):
        st.markdown(
            """
            - Answer based on how you usually are, not how you'd like to be.
            - There are no right or wrong answers, and no time limit.
            - **Your answers are not saved.** Nothing is stored, sent or shared; closing the tab erases everything.
            - This version is for people aged **16 and over**. Children and teenagers need
              different questionnaires, completed with a parent and a professional.
            """
        )

    age_ok = st.checkbox("I am 16 or older")
    understood = st.checkbox("I understand this is a screening tool, not a diagnosis")
    st.button("Start", type="primary", disabled=not (age_ok and understood), on_click=go, args=("questions",))

# ------------------------------------------------------------ questions
elif st.session_state.stage == "questions":
    st.title("The questions")
    st.caption("Choose the answer that fits you best for each statement.")

    answered = sum(st.session_state.get(f"q{i}") is not None for i in range(1, NUM_ITEMS + 1))
    st.progress(answered / NUM_ITEMS, text=f"{answered} of {NUM_ITEMS} answered")

    for i, question in enumerate(qs.questions, start=1):
        st.radio(f"**{i}.** {question}", ANSWER_OPTIONS, index=None, key=f"q{i}")
        if i < NUM_ITEMS:
            st.divider()

    answers = [st.session_state.get(f"q{i}") for i in range(1, NUM_ITEMS + 1)]
    complete = all(a is not None for a in answers)

    col1, col2 = st.columns([1, 1])
    col1.button("Back", on_click=go, args=("intro",))
    if col2.button("See my result", type="primary", disabled=not complete, use_container_width=True):
        st.session_state.score = total_score(answers)
        go("results")
        st.rerun()
    if not complete:
        st.caption("Answer all 10 questions to see your result.")

# -------------------------------------------------------------- results
elif st.session_state.stage == "results":
    score = st.session_state.score
    st.title("Your result")

    st.markdown(f'<div class="score">{score} / {NUM_ITEMS}</div>', unsafe_allow_html=True)
    st.progress(score / NUM_ITEMS)
    st.markdown(f'<p class="muted">A score of {CUTOFF} or more is the point where a full assessment is usually suggested.</p>',
                unsafe_allow_html=True)

    if score >= CUTOFF:
        st.markdown(
            f"""
            Your score is **at or above {CUTOFF}**. This means some of your answers are
            similar to those often given by autistic people, and **a full assessment by a
            qualified professional could be worthwhile.**

            This is **not a diagnosis**. Many things can raise a score, including anxiety,
            ADHD, depression, or simply your personality. Plenty of people who score
            above {CUTOFF} are not autistic, and an assessment is the way to find out.
            """
        )
    else:
        st.markdown(
            f"""
            Your score is **below {CUTOFF}**. On this short questionnaire, your answers
            don't point towards a full assessment.

            **This does not rule out autism.** Short questionnaires can miss autistic
            people, especially women and people who have learned to mask or camouflage
            their traits. If you still have concerns, it's completely reasonable to
            speak to a professional.
            """
        )

    st.subheader("What you can do next")
    st.markdown(
        """
        - **Talk to a professional.** An adult autism assessment is done by a
          **psychiatrist** or a **clinical psychologist** (in India, look for one registered
          with the Rehabilitation Council of India). Your GP or family doctor can refer you.
        - **Public options in India:** psychiatry departments at government medical
          colleges and district hospitals, and national institutes such as **NIMHANS**
          (Bengaluru) and **AIIMS**, offer assessments, usually at lower cost.
        - **Prepare for the appointment:** note examples from daily life and childhood
          (social situations, routines, sensory experiences). If possible, ask a parent or
          someone who knew you as a child what you were like.
        - **If you're feeling overwhelmed or distressed**, you can call **Tele-MANAS**,
          India's free 24/7 mental health helpline: **14416** or **1-800-891-4416**.
        """
    )

    with st.expander("About this test"):
        st.markdown(
            f"""
            - The **AQ-10 (Adult)** was developed by the Autism Research Centre, University
              of Cambridge (Allison, Auyeung & Baron-Cohen, 2012), as a quick way to decide
              who might benefit from a full autism assessment.
            - Each answer scores 0 or 1, giving a total out of 10. The suggested referral
              point is {CUTOFF}.
            - It is a **screening** tool. It cannot diagnose autism or any other condition.
            - This app does not store, send or share your answers.
            """
        )

    st.button("Take it again", on_click=restart)

# --------------------------------------------------------------- footer
st.divider()
st.markdown(
    f'<p class="muted">Built by Aastha Mahajan as an open-source project · '
    f'<a href="{REPO_URL}" target="_blank">Source code</a> · Not medical advice.</p>',
    unsafe_allow_html=True,
)

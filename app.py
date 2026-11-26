import sqlite3
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Job Application Organizer",
    page_icon="🧭",
    layout="wide",
)

DB_PATH = Path(__file__).parent / "applications.db"
STATUSES = [
    "Saved", "Applied", "Assessment", "Interview",
    "Offer", "Rejected", "Withdrawn"
]


def connect_db():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            role TEXT NOT NULL,
            location TEXT DEFAULT '',
            job_url TEXT DEFAULT '',
            status TEXT NOT NULL,
            applied_on TEXT,
            follow_up TEXT,
            salary TEXT DEFAULT '',
            contact TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    connection.commit()
    return connection


def get_applications():
    return pd.read_sql_query(
        """
        SELECT * FROM applications
        ORDER BY COALESCE(applied_on, created_at) DESC, id DESC
        """,
        connect_db(),
    )


def add_application(values):
    connection = connect_db()
    connection.execute(
        """
        INSERT INTO applications
        (company, role, location, job_url, status, applied_on, follow_up,
         salary, contact, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (*values, datetime.now().isoformat(timespec="seconds")),
    )
    connection.commit()


def change_status(application_id, status):
    connection = connect_db()
    connection.execute(
        "UPDATE applications SET status = ? WHERE id = ?",
        (status, application_id),
    )
    connection.commit()


def remove_application(application_id):
    connection = connect_db()
    connection.execute(
        "DELETE FROM applications WHERE id = ?",
        (application_id,),
    )
    connection.commit()


def parse_date(value):
    if not value or pd.isna(value):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


st.title("🧭 Job Application Organizer")
st.caption(
    "Track applications, interviews, and follow-up dates in one place."
)

with st.sidebar:
    st.header("Add an application")

    with st.form("add_application_form", clear_on_submit=True):
        company = st.text_input("Company *")
        role = st.text_input("Job title *")
        location = st.text_input("Location or remote")
        job_url = st.text_input("Job posting link")
        status = st.selectbox("Status", STATUSES, index=1)
        applied_on = st.date_input("Applied date", value=date.today())
        follow_up = st.date_input("Follow-up date (optional)", value=None)
        salary = st.text_input("Salary or range (optional)")
        contact = st.text_input("Contact (optional)")
        notes = st.text_area("Notes (optional)")

        save = st.form_submit_button(
            "Save application",
            type="primary",
            use_container_width=True,
        )

    if save:
        if not company.strip() or not role.strip():
            st.error("Company and job title are required.")
        else:
            add_application((
                company.strip(),
                role.strip(),
                location.strip(),
                job_url.strip(),
                status,
                applied_on.isoformat(),
                follow_up.isoformat() if follow_up else None,
                salary.strip(),
                contact.strip(),
                notes.strip(),
            ))
            st.success("Application saved.")
            st.rerun()


applications = get_applications()

if applications.empty:
    st.info("No applications saved yet. Add one using the form in the sidebar.")
else:
    today = date.today()
    due_mask = applications["follow_up"].apply(
        lambda value: parse_date(value) is not None
        and parse_date(value) <= today
    )
    active = applications[
        ~applications["status"].isin(["Rejected", "Withdrawn"])
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applications", len(applications))
    col2.metric("Active", len(active))
    col3.metric("Interviews", int((applications["status"] == "Interview").sum()))
    col4.metric("Follow-ups due", int(due_mask.sum()))

    due = applications[
        due_mask & ~applications["status"].isin(["Rejected", "Withdrawn"])
    ]
    if not due.empty:
        reminders = [
            f"{row.company} — {row.role}"
            for row in due.itertuples(index=False)
        ]
        st.warning("Follow-up date reached: " + "; ".join(reminders))

    st.subheader("Your applications")

    search_col, status_col, sort_col = st.columns([2, 1, 1])
    search = search_col.text_input(
        "Search",
        placeholder="Search company, role, location, or notes",
    )
    status_filter = status_col.multiselect("Filter by status", STATUSES)
    sort_by = sort_col.selectbox(
        "Sort",
        ["Newest first", "Follow-up date", "Company A–Z"],
    )

    shown = applications.copy()

    if search:
        searchable = shown[
            ["company", "role", "location", "notes"]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        shown = shown[
            searchable.str.contains(search, case=False, regex=False)
        ]

    if status_filter:
        shown = shown[shown["status"].isin(status_filter)]

    if sort_by == "Follow-up date":
        shown = shown.sort_values("follow_up", na_position="last")
    elif sort_by == "Company A–Z":
        shown = shown.sort_values(
            "company", key=lambda values: values.str.lower()
        )
    else:
        shown = shown.sort_values(
            ["applied_on", "id"],
            ascending=False,
            na_position="last",
        )

    table = shown[
        [
            "company", "role", "location", "status", "applied_on",
            "follow_up", "salary", "job_url", "contact", "notes",
        ]
    ].rename(columns={
        "company": "Company",
        "role": "Job title",
        "location": "Location",
        "status": "Status",
        "applied_on": "Applied",
        "follow_up": "Follow-up",
        "salary": "Salary",
        "job_url": "Job link",
        "contact": "Contact",
        "notes": "Notes",
    })

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Job link": st.column_config.LinkColumn("Job link"),
            "Notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )

    with st.expander("Update or delete an application"):
        choices = {
            f"{row.company} — {row.role} (ID {row.id})": int(row.id)
            for row in applications.itertuples(index=False)
        }
        selected = st.selectbox("Choose an application", list(choices))
        application_id = choices[selected]
        current_status = applications.loc[
            applications["id"] == application_id, "status"
        ].iloc[0]

        edit_col, delete_col = st.columns(2)

        new_status = edit_col.selectbox(
            "New status",
            STATUSES,
            index=STATUSES.index(current_status),
        )
        if edit_col.button("Update status"):
            change_status(application_id, new_status)
            st.success("Status updated.")
            st.rerun()

        confirm_delete = delete_col.checkbox("Confirm deletion")
        if delete_col.button("Delete application", disabled=not confirm_delete):
            remove_application(application_id)
            st.rerun()

    st.subheader("Application pipeline")
    counts = (
        applications.groupby("status")
        .size()
        .reindex(STATUSES, fill_value=0)
    )
    st.bar_chart(counts)

    csv_data = applications.drop(
        columns=["id", "created_at"], errors="ignore"
    ).to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Download applications as CSV",
        data=csv_data,
        file_name="job_applications.csv",
        mime="text/csv",
    )

st.caption(
    "This MVP stores data in a local SQLite file. On free hosted services, "
    "local data may not survive a restart or redeployment. Download CSV backups regularly."
)

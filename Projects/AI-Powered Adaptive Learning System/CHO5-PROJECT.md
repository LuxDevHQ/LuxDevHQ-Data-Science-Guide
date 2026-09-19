# AI-Powered Adaptive Learning System
### A Beginner-Friendly Project Guide | Groq + Aiven PostgreSQL + FastAPI

---

## 1. Project Overview

You are going to build a learning platform that adjusts to each student. The system will remember what a student got wrong, ask an AI to help explain it, and store everything in a real cloud database.

Before writing any code, think about these questions:

- What information do you need to store about a student?
- What should happen the moment a student submits a wrong answer?
- How would a teacher know which student needs help?

---

## 2. Technology Stack

You will use four tools. Research each one before you start.

| Layer | Tool | What to find out |
|---|---|---|
| Backend | FastAPI | How do you create a route that accepts data? |
| AI | Groq API | How do you send a prompt and read the response? |
| Database | Aiven PostgreSQL | How do you connect using a connection string? |
| Analytics | SQL | What does GROUP BY and HAVING do? |

---

## 3. Database Design

You need to design tables before writing any code. Think about what data each table should hold.

### Table 1: Students

Think about what basic information identifies a student. Every table needs a way to tell rows apart from each other. What column makes each student unique?

Hint: think about id, name, email, and when they joined.

### Table 2: Quiz Results

Every time a student answers a question, you want to save that moment. What details matter about that moment?

Hint: think about which student, which topic, what they answered, whether it was right, how long it took, and what the AI said back.

### Table 3: Performance History

This table is a summary. It does not store individual answers. It stores a rolled-up picture of how a student is doing over time.

Hint: think about averages and totals per topic per student.

---

## 4. Setting Up Your Project

### Folder Structure

Before writing code, plan your files. A good structure separates concerns: one file for database, one for AI, one for routes.

Hint: think about what belongs together. Database connection code should not live in the same file as your AI calls.

### Environment Variables

You should never put passwords or API keys directly in your code. Use a separate file to store secrets.

Hint: look up what a `.env` file is and how `python-dotenv` reads it. Your database URL and Groq key go here.

### Packages to Install

You need four packages. Think about what each one does before installing it.

Hint: one for your web server, one for your database connection, one for Groq, one for reading your `.env` file.

---

## 5. Building the Backend

### Connecting to Aiven PostgreSQL

Your database connection needs the URL from your Aiven dashboard. The URL contains your host, port, username, password, and database name all in one string.

Hint: look up how `asyncpg.connect()` works. Think about what happens if the connection fails.

### Endpoint 1: Submit an Answer

This is the most important endpoint. When a student submits an answer, several things need to happen in the right order.

Think about the order:

1. Check if the answer is correct
2. Ask the AI for feedback
3. Save everything to the database
4. Return the result to the student

Hint: the endpoint receives data from the student. What shape should that data be? Look up Pydantic `BaseModel`.

### Endpoint 2: Student Progress

A teacher or student should be able to see how a student is performing across all topics.

Hint: this endpoint reads from the database, not writes. Think about which SQL keywords group results by topic and calculate averages.

---

## 6. Connecting Groq AI

### How Groq Works

Groq takes a text prompt and returns a text response. Your job is to write a good prompt that gives Groq enough context to respond helpfully.

Hint: think about what information Groq needs. It cannot see your database. You have to pass the topic, the question, the student's answer, and the correct answer inside the prompt text.

### Writing Two Different Prompts

The feedback should be different depending on whether the student got it right or wrong.

For a wrong answer, think about what a good teacher would do:
- Would they just say "wrong"?
- Or would they explain, show an example, and give another chance?

For a correct answer, think about what keeps a student motivated and challenged.

Hint: use an if/else to build different prompts for each case.

### Generating New Questions

You also want Groq to generate quiz questions on any topic at any difficulty level.

Hint: think about what information you need to pass in, and what format you want Groq to respond in. You may want to tell Groq exactly how to structure its response.

---

## 7. SQL Analytics

This is where your system becomes useful for teachers. You will write queries that summarize student performance.

### Query 1: Find Struggling Students

You want to find students whose average score is below a certain threshold.

Hint: you need to join two tables, group by student, calculate an average, and then filter groups using a keyword that filters after grouping (not WHERE).

### Query 2: Find the Hardest Topic

You want to find which topic has the lowest average score across all students.

Hint: group by topic, calculate the average score, and sort the results so the lowest comes first.

### Query 3: Individual Student Report

A student wants to see their own performance broken down by topic.

Hint: filter by student, group by topic, and show the average score and average time taken.

### Exposing Analytics as an Endpoint

Once your queries work, wrap them in a FastAPI endpoint so teachers can access the data through the API.

Hint: think about what a teacher would want to pass in as a parameter. Maybe a score threshold? Maybe a student id?

---

## 8. Real-Time Feedback Flow

This is the full loop. When it all works together, the sequence looks like this:

```
Student submits answer
        |
        v
Backend checks: correct or incorrect?
        |
        v
Build a prompt using the context
        |
        v
Send prompt to Groq
        |
        v
Groq returns feedback
        |
        v
Save result + feedback to PostgreSQL
        |
        v
Return feedback to student instantly
```

Think about what could go wrong at each step. What happens if Groq is slow? What happens if the database save fails? Good code handles these cases.

---

## 9. Project Phases

Work through this one phase at a time. Do not move to the next phase until the current one works.

| Phase | Focus | You know it works when... |
|---|---|---|
| 1 | Database + API | You can submit an answer and see it saved in Aiven |
| 2 | Groq Integration | Wrong answers return a helpful AI explanation |
| 3 | Analytics | You can query which students are struggling |
| 4 | Real-Time Feedback | All steps happen in one request, end to end |

---

## 10. Questions to Guide Your Thinking

Use these questions as checkpoints as you build:

- Why do we store ai_feedback in the database instead of just showing it once?
- What would happen if two students submit answers at the exact same time?
- How would you change the system to support multiple teachers?
- What SQL query would tell you which student has improved the most over time?
- Why should secrets like API keys never be committed to GitHub?

---

## 11. Running Your Application

Once your code is written, you need to start the server and test it.

Hint: FastAPI uses a command-line tool called `uvicorn` to run. Look up the basic command. FastAPI also automatically creates a page where you can test your endpoints in the browser without writing any extra code.

Think about: how would you test your `/submit-answer` endpoint before building a frontend?

---

*Build it phase by phase. Test each piece before moving forward. The goal is not perfect code — it is a working system you understand.*


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ============================================================
# DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE
# Extended Major Project Dashboard
# ============================================================

st.set_page_config(
    page_title="Social Engagement Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
.main {background-color:#f7f8fc;}
.block-container {padding-top:1.2rem; padding-bottom:2rem;}
[data-testid="stMetric"] {
    background:white;
    border:1px solid #e5e7eb;
    padding:18px;
    border-radius:14px;
    box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.dashboard-card {
    background:white;
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:18px;
    margin-bottom:14px;
}
.small-note {color:#64748b;font-size:0.88rem;}
.big-title {font-size:2.1rem;font-weight:800;}
.section-title {font-size:1.35rem;font-weight:750;margin-top:10px;}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
BASE = __file__.rsplit("/", 1)[0] if "/" in __file__ else __file__.rsplit("\\", 1)[0]
DATA_PATH = BASE + "/data/sample_social_media_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [
        "reach","likes","comments","shares","saves",
        "retention_rate","follower_growth","followers",
        "video_length_sec"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["engagement"] = df["likes"] + df["comments"] + df["shares"] + df["saves"]
    df["engagement_rate"] = np.where(
        df["reach"] > 0, df["engagement"] / df["reach"] * 100, 0
    )
    df["viral_score"] = np.where(
        df["reach"] > 0,
        (4*df["shares"] + 3*df["saves"] + 2*df["comments"] + 0.5*df["likes"]) / df["reach"],
        0
    )
    df["save_share_ratio"] = np.where(df["shares"] > 0, df["saves"]/df["shares"], 0)
    df["interaction_rate"] = np.where(
        df["reach"] > 0,
        (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / df["reach"] * 100,
        0
    )

    positive_words = [
        "help","same","relate","relatable","feel","felt","true",
        "exactly","me","understand","anxiety","stress","alone",
        "struggle","struggling","love","useful"
    ]
    negative_words = [
        "hate","bad","worst","angry","boring","fake","wrong","problem"
    ]

    def polarity(text):
        text = str(text).lower()
        pos = sum(w in text for w in positive_words)
        neg = sum(w in text for w in negative_words)
        if pos > neg:
            return min(1.0, 0.15 * pos)
        if neg > pos:
            return max(-1.0, -0.15 * neg)
        return 0.0

    df["polarity"] = df["comment_text"].apply(polarity)
    df["sentiment"] = np.select(
        [df["polarity"] > 0.05, df["polarity"] < -0.05],
        ["Positive", "Negative"],
        default="Neutral"
    )
    relatable_terms = [
        "same","relate","relatable","me","feel","felt","anxiety",
        "stress","alone","struggle","exactly","understand"
    ]
    df["relatable"] = df["comment_text"].astype(str).str.lower().apply(
        lambda x: "Relatable" if any(t in x for t in relatable_terms) else "Neutral"
    )
    df["problem_awareness"] = df["comment_text"].astype(str).str.lower().apply(
        lambda x: "High" if any(t in x for t in [
            "problem","struggle","anxiety","stress","fear","alone","difficult"
        ]) else "Normal"
    )
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.code(str(e))
    st.info("Make sure data/sample_social_media_data.csv exists inside the project folder.")
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("# 📊 Social Intelligence")
st.sidebar.caption("Data-Driven Social Engagement Initiative")

pages = [
    "🏠 Executive Dashboard",
    "📈 Performance Analytics",
    "🚀 Virality Intelligence",
    "💬 Sentiment & Relatability",
    "🧪 A/B Testing Lab",
    "👥 Audience Intelligence",
    "🎯 Content Optimization",
    "🔮 Trend Forecasting",
    "📊 Growth & KPI Center",
    "🔎 Data Explorer",
    "📑 Project Insights"
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")

topics = ["All"] + sorted(df["topic"].dropna().unique().tolist())
formats = ["All"] + sorted(df["format"].dropna().unique().tolist())
hooks = ["All"] + sorted(df["hook"].dropna().unique().tolist())

topic_filter = st.sidebar.selectbox("Topic", topics)
format_filter = st.sidebar.selectbox("Format", formats)
hook_filter = st.sidebar.selectbox("Hook", hooks)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))

filtered = df.copy()

if topic_filter != "All":
    filtered = filtered[filtered["topic"] == topic_filter]
if format_filter != "All":
    filtered = filtered[filtered["format"] == format_filter]
if hook_filter != "All":
    filtered = filtered[filtered["hook"] == hook_filter]

if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered["date"].dt.date >= date_range[0]) &
        (filtered["date"].dt.date <= date_range[1])
    ]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -------------------- HELPERS --------------------
def moneyless_metric(label, value, suffix=""):
    st.metric(label, f"{value:,.2f}{suffix}")

def metric_row(data):
    reach = data["reach"].sum()
    eng = data["engagement"].sum()
    viral = data["viral_score"].mean()
    rel = (data["relatable"] == "Relatable").mean() * 100
    growth = data["follower_growth"].sum()
    retention = data["retention_rate"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Reach", f"{reach:,.0f}")
    c2.metric("Engagements", f"{eng:,.0f}")
    c3.metric("Avg Viral Score", f"{viral:.3f}")
    c4.metric("Relatability", f"{rel:.1f}%")
    c5.metric("Follower Growth", f"{growth:,.0f}")
    c6.metric("Avg Retention", f"{retention:.1f}%")

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-note">{subtitle}</div>', unsafe_allow_html=True)

def bar_chart(data, x, y, title, color=None):
    fig = px.bar(data, x=x, y=y, title=title, color=color, text_auto=".2f")
    fig.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig, use_container_width=True)

def line_chart(data, x, y, title, color=None):
    fig = px.line(data, x=x, y=y, title=title, color=color, markers=True)
    fig.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 1. EXECUTIVE DASHBOARD
# ============================================================
if page == "🏠 Executive Dashboard":
    st.markdown('<div class="big-title">📊 Social Engagement Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.caption("Executive view of content performance, audience emotion, virality and growth.")

    metric_row(filtered)

    st.markdown("---")
    section("Performance Overview", "High-level signals for the selected period.")

    col1, col2 = st.columns(2)

    with col1:
        topic_perf = filtered.groupby("topic", as_index=False).agg(
            Reach=("reach","sum"),
            Engagement=("engagement","sum"),
            Viral_Score=("viral_score","mean")
        ).sort_values("Engagement", ascending=False)
        bar_chart(topic_perf, "topic", "Engagement", "Engagement by Topic")

    with col2:
        sent = filtered["sentiment"].value_counts().reset_index()
        sent.columns = ["Sentiment","Count"]
        fig = px.pie(sent, names="Sentiment", values="Count", hole=.45,
                     title="Audience Sentiment Distribution")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        daily = filtered.groupby("date", as_index=False).agg(
            Engagement=("engagement","sum"),
            Reach=("reach","sum")
        )
        fig = px.line(daily, x="date", y=["Engagement","Reach"],
                      title="Daily Reach & Engagement", markers=True)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        growth = filtered.groupby("date", as_index=False)["follower_growth"].sum()
        line_chart(growth, "date", "follower_growth", "Daily Follower Growth")

    section("Top Performing Content")
    top = filtered.sort_values(["viral_score","engagement"], ascending=False).head(10)
    st.dataframe(
        top[[
            "content_id","date","topic","format","hook",
            "reach","likes","comments","shares","saves",
            "engagement_rate","viral_score"
        ]],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 2. PERFORMANCE ANALYTICS
# ============================================================
elif page == "📈 Performance Analytics":
    st.markdown('<div class="big-title">📈 Content Performance Analytics</div>', unsafe_allow_html=True)
    st.caption("Detailed analysis of reach, engagement, retention and content formats.")

    metric_row(filtered)

    section("Format Performance")
    format_perf = filtered.groupby("format", as_index=False).agg(
        Reach=("reach","sum"),
        Engagement=("engagement","mean"),
        Engagement_Rate=("engagement_rate","mean"),
        Retention=("retention_rate","mean"),
        Follower_Growth=("follower_growth","mean")
    )

    c1,c2 = st.columns(2)
    with c1:
        bar_chart(format_perf, "format", "Engagement", "Average Engagement by Format")
    with c2:
        bar_chart(format_perf, "format", "Retention", "Retention by Format")

    section("Topic Performance Matrix")
    topic_perf = filtered.groupby("topic", as_index=False).agg(
        Reach=("reach","sum"),
        Likes=("likes","sum"),
        Comments=("comments","sum"),
        Shares=("shares","sum"),
        Saves=("saves","sum"),
        Engagement=("engagement","sum"),
        Viral_Score=("viral_score","mean"),
        Retention=("retention_rate","mean"),
        Growth=("follower_growth","sum")
    )
    st.dataframe(topic_perf.round(2), use_container_width=True, hide_index=True)

    section("Reach vs Engagement")
    fig = px.scatter(
        filtered, x="reach", y="engagement",
        size="viral_score", color="topic",
        hover_data=["content_id","format","hook"],
        title="Reach vs Engagement — Bubble Size = Viral Score"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    section("Video Length Analysis")
    bins = [0,15,30,60,90,180,9999]
    labels = ["0–15s","16–30s","31–60s","61–90s","91–180s","180s+"]
    temp = filtered.copy()
    temp["length_group"] = pd.cut(temp["video_length_sec"], bins=bins, labels=labels, include_lowest=True)
    length_perf = temp.groupby("length_group", observed=False, as_index=False).agg(
        Engagement=("engagement","mean"),
        Retention=("retention_rate","mean"),
        Viral_Score=("viral_score","mean")
    )
    bar_chart(length_perf, "length_group", "Engagement", "Average Engagement by Video Length")

# ============================================================
# 3. VIRALITY
# ============================================================
elif page == "🚀 Virality Intelligence":
    st.markdown('<div class="big-title">🚀 Virality Prediction & Intelligence</div>', unsafe_allow_html=True)
    st.caption("Weighted virality model emphasizing shares and saves.")

    st.info(
        "Viral Score = (4 × Shares + 3 × Saves + 2 × Comments + 0.5 × Likes) / Reach"
    )

    metric_row(filtered)

    c1,c2,c3 = st.columns(3)
    c1.metric("Highest Viral Score", f"{filtered['viral_score'].max():.3f}")
    c2.metric("Avg Share Count", f"{filtered['shares'].mean():.1f}")
    c3.metric("Avg Save Count", f"{filtered['saves'].mean():.1f}")

    section("Viral Score by Topic")
    viral_topic = filtered.groupby("topic", as_index=False).agg(
        Viral_Score=("viral_score","mean"),
        Shares=("shares","mean"),
        Saves=("saves","mean")
    ).sort_values("Viral_Score", ascending=False)
    bar_chart(viral_topic, "topic", "Viral_Score", "Average Viral Score by Topic")

    section("Share–Save Relationship")
    fig = px.scatter(
        filtered, x="shares", y="saves", color="topic",
        size="viral_score", hover_data=["content_id","reach"],
        title="Shares vs Saves — High values indicate stronger organic potential"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    section("Most Viral Posts")
    viral_posts = filtered.sort_values("viral_score", ascending=False).head(15)
    st.dataframe(
        viral_posts[[
            "content_id","date","topic","format","reach",
            "shares","saves","comments","likes","viral_score"
        ]].round(3),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 4. SENTIMENT
# ============================================================
elif page == "💬 Sentiment & Relatability":
    st.markdown('<div class="big-title">💬 Audience Sentiment & Relatability</div>', unsafe_allow_html=True)
    st.caption("NLP-style comment analysis for emotional resonance and problem awareness.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Comments Analyzed", f"{len(filtered):,}")
    c2.metric("Relatable Comments", f"{(filtered['relatable']=='Relatable').sum():,}")
    c3.metric("Relatability Rate", f"{(filtered['relatable']=='Relatable').mean()*100:.1f}%")
    c4.metric("High Problem Awareness", f"{(filtered['problem_awareness']=='High').mean()*100:.1f}%")

    col1,col2 = st.columns(2)
    with col1:
        sent = filtered["sentiment"].value_counts().reset_index()
        sent.columns = ["Sentiment","Count"]
        fig = px.pie(sent, names="Sentiment", values="Count",
                     hole=.45, title="Sentiment Mix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rel = filtered["relatable"].value_counts().reset_index()
        rel.columns = ["Category","Count"]
        fig = px.bar(rel, x="Category", y="Count", title="Relatable vs Neutral")
        st.plotly_chart(fig, use_container_width=True)

    section("Sentiment by Topic")
    topic_sent = pd.crosstab(filtered["topic"], filtered["sentiment"]).reset_index()
    fig = px.bar(topic_sent, x="topic", y=topic_sent.columns[1:],
                 title="Sentiment Distribution Across Topics")
    st.plotly_chart(fig, use_container_width=True)

    section("Comment Intelligence")
    comments = filtered[[
        "content_id","topic","comment_text","sentiment",
        "relatable","problem_awareness","polarity"
    ]].sort_values("polarity", ascending=False)
    st.dataframe(comments, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download NLP Analysis CSV",
        comments.to_csv(index=False),
        "sentiment_relatability_analysis.csv",
        "text/csv"
    )

# ============================================================
# 5. A/B TESTING
# ============================================================
elif page == "🧪 A/B Testing Lab":
    st.markdown('<div class="big-title">🧪 A/B Testing & Experimentation Lab</div>', unsafe_allow_html=True)
    st.caption("Compare formats, hooks, caption styles and topics using engagement outcomes.")

    dimension = st.selectbox(
        "Experiment dimension",
        ["format","hook","caption_style","topic"]
    )

    ab = filtered.groupby(dimension, as_index=False).agg(
        Posts=("content_id","count"),
        Avg_Engagement=("engagement","mean"),
        Avg_Engagement_Rate=("engagement_rate","mean"),
        Avg_Viral_Score=("viral_score","mean"),
        Avg_Retention=("retention_rate","mean"),
        Avg_Follower_Growth=("follower_growth","mean")
    ).sort_values("Avg_Engagement", ascending=False)

    best = ab.iloc[0]
    st.success(
        f"Best {dimension.replace('_',' ').title()}: **{best[dimension]}** "
        f"with average engagement of **{best['Avg_Engagement']:.1f}**."
    )

    c1,c2 = st.columns(2)
    with c1:
        bar_chart(ab, dimension, "Avg_Engagement", f"Engagement by {dimension}")
    with c2:
        bar_chart(ab, dimension, "Avg_Viral_Score", f"Viral Score by {dimension}")

    section("A/B Test Scorecard")
    st.dataframe(ab.round(3), use_container_width=True, hide_index=True)

    section("Retention Comparison")
    fig = px.bar(ab, x=dimension, y="Avg_Retention",
                 title="Average Retention by Experiment Group",
                 text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. AUDIENCE INTELLIGENCE
# ============================================================
elif page == "👥 Audience Intelligence":
    st.markdown('<div class="big-title">👥 Audience Intelligence</div>', unsafe_allow_html=True)
    st.caption("Understand which topics generate interaction, emotion and growth.")

    topic = filtered.groupby("topic", as_index=False).agg(
        Posts=("content_id","count"),
        Reach=("reach","sum"),
        Engagement=("engagement","sum"),
        Shares=("shares","sum"),
        Saves=("saves","sum"),
        Comments=("comments","sum"),
        Follower_Growth=("follower_growth","sum"),
        Relatability=("relatable", lambda x: (x=="Relatable").mean()*100)
    )

    section("Audience Topic Ranking")
    topic["Audience_Score"] = (
        topic["Engagement"].rank(pct=True) * 0.30 +
        topic["Shares"].rank(pct=True) * 0.25 +
        topic["Saves"].rank(pct=True) * 0.20 +
        topic["Relatability"].rank(pct=True) * 0.25
    ) * 100
    topic = topic.sort_values("Audience_Score", ascending=False)

    c1,c2 = st.columns(2)
    with c1:
        bar_chart(topic, "topic", "Audience_Score", "Audience Opportunity Score")
    with c2:
        bar_chart(topic, "topic", "Relatability", "Relatability by Topic")

    section("Engagement Composition")
    comp = filtered[["likes","comments","shares","saves"]].sum().reset_index()
    comp.columns = ["Metric","Value"]
    fig = px.pie(comp, names="Metric", values="Value", hole=.4,
                 title="Total Engagement Composition")
    st.plotly_chart(fig, use_container_width=True)

    section("Topic Intelligence Table")
    st.dataframe(topic.round(2), use_container_width=True, hide_index=True)

# ============================================================
# 7. OPTIMIZATION
# ============================================================
elif page == "🎯 Content Optimization":
    st.markdown('<div class="big-title">🎯 Engagement Optimization Recommender</div>', unsafe_allow_html=True)
    st.caption("Recommendations derived from historical performance in the selected dataset.")

    def best_value(column, metric="engagement"):
        g = filtered.groupby(column)[metric].mean().sort_values(ascending=False)
        return g.index[0], g.iloc[0], g

    topic_best, topic_val, topic_all = best_value("topic")
    format_best, format_val, format_all = best_value("format")
    hook_best, hook_val, hook_all = best_value("hook")
    caption_best, caption_val, caption_all = best_value("caption_style")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Topic", str(topic_best))
    c2.metric("Best Format", str(format_best))
    c3.metric("Best Hook", str(hook_best))
    c4.metric("Best Caption", str(caption_best))

    section("Recommended Content Formula")
    st.markdown(f"""
    <div class="dashboard-card">
    <b>Recommended Topic:</b> {topic_best}<br><br>
    <b>Recommended Format:</b> {format_best}<br><br>
    <b>Recommended Hook:</b> {hook_best}<br><br>
    <b>Recommended Caption Style:</b> {caption_best}<br><br>
    <b>Suggested Strategy:</b> Prioritize the combinations that historically produced
    stronger engagement, shares, saves and retention.
    </div>
    """, unsafe_allow_html=True)

    section("Topic Recommendation Ranking")
    topic_rank = filtered.groupby("topic", as_index=False).agg(
        Engagement=("engagement","mean"),
        Viral_Score=("viral_score","mean"),
        Retention=("retention_rate","mean"),
        Growth=("follower_growth","mean"),
        Relatability=("relatable", lambda x:(x=="Relatable").mean()*100)
    )
    topic_rank["Recommendation_Score"] = (
        topic_rank["Engagement"].rank(pct=True)*0.30 +
        topic_rank["Viral_Score"].rank(pct=True)*0.25 +
        topic_rank["Retention"].rank(pct=True)*0.15 +
        topic_rank["Growth"].rank(pct=True)*0.15 +
        topic_rank["Relatability"].rank(pct=True)*0.15
    )*100
    topic_rank = topic_rank.sort_values("Recommendation_Score", ascending=False)
    st.dataframe(topic_rank.round(2), use_container_width=True, hide_index=True)

    section("Best Performing Combinations")
    combo = filtered.groupby(
        ["topic","format","hook"], as_index=False
    ).agg(
        Posts=("content_id","count"),
        Engagement=("engagement","mean"),
        Viral_Score=("viral_score","mean"),
        Retention=("retention_rate","mean")
    ).sort_values(["Engagement","Viral_Score"], ascending=False)
    st.dataframe(combo.head(20).round(3), use_container_width=True, hide_index=True)

# ============================================================
# 8. FORECASTING
# ============================================================
elif page == "🔮 Trend Forecasting":
    st.markdown('<div class="big-title">🔮 Trend Forecasting & Growth Projection</div>', unsafe_allow_html=True)
    st.caption("Simple linear forecasting based on historical follower trajectory.")

    daily = filtered.groupby("date", as_index=False).agg(
        Followers=("followers","max"),
        Growth=("follower_growth","sum"),
        Engagement=("engagement","sum")
    ).sort_values("date")

    if len(daily) >= 3:
        daily["day_index"] = np.arange(len(daily))
        model = LinearRegression()
        model.fit(daily[["day_index"]], daily["Followers"])

        future_days = 14
        future_idx = np.arange(len(daily), len(daily)+future_days)
        pred = model.predict(future_idx.reshape(-1,1))
        future_dates = pd.date_range(
            daily["date"].max() + pd.Timedelta(days=1),
            periods=future_days
        )

        forecast = pd.DataFrame({
            "date": future_dates,
            "followers": pred,
            "type": "Forecast"
        })
        history = daily[["date","Followers"]].copy()
        history["type"] = "Historical"
        history.columns = ["date","followers","type"]

        combined = pd.concat([history, forecast], ignore_index=True)

        fig = px.line(
            combined, x="date", y="followers", color="type",
            markers=True, title="Historical + 14-Day Follower Forecast"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Current Followers", f"{daily['Followers'].iloc[-1]:,.0f}")
        c2.metric("Forecast Followers", f"{pred[-1]:,.0f}")
        c3.metric("Projected Change", f"{pred[-1]-daily['Followers'].iloc[-1]:+,.0f}")

        section("Daily Growth Trend")
        line_chart(daily, "date", "Growth", "Follower Growth Trend")
    else:
        st.warning("Not enough historical records for forecasting.")

    section("Topic Trend Potential")
    trend = filtered.groupby("topic", as_index=False).agg(
        Engagement=("engagement","mean"),
        Shares=("shares","mean"),
        Saves=("saves","mean"),
        Growth=("follower_growth","mean"),
        Viral_Score=("viral_score","mean")
    )
    trend["Trend_Potential"] = (
        trend["Shares"].rank(pct=True)*.25 +
        trend["Saves"].rank(pct=True)*.20 +
        trend["Growth"].rank(pct=True)*.25 +
        trend["Viral_Score"].rank(pct=True)*.30
    )*100
    trend = trend.sort_values("Trend_Potential", ascending=False)
    bar_chart(trend, "topic", "Trend_Potential", "Topic Trend Potential")

# ============================================================
# 9. KPI CENTER
# ============================================================
elif page == "📊 Growth & KPI Center":
    st.markdown('<div class="big-title">📊 Growth & KPI Command Center</div>', unsafe_allow_html=True)
    st.caption("Track the key indicators that explain audience and content growth.")

    reach = filtered["reach"].sum()
    engagement = filtered["engagement"].sum()
    shares = filtered["shares"].sum()
    saves = filtered["saves"].sum()
    comments = filtered["comments"].sum()
    likes = filtered["likes"].sum()
    growth = filtered["follower_growth"].sum()

    kpis = pd.DataFrame({
        "KPI": [
            "Reach","Likes","Comments","Shares","Saves",
            "Engagement","Follower Growth"
        ],
        "Value": [
            reach,likes,comments,shares,saves,engagement,growth
        ]
    })

    section("Core KPI Cards")
    c = st.columns(4)
    c[0].metric("Reach", f"{reach:,.0f}")
    c[1].metric("Likes", f"{likes:,.0f}")
    c[2].metric("Comments", f"{comments:,.0f}")
    c[3].metric("Shares", f"{shares:,.0f}")

    c = st.columns(4)
    c[0].metric("Saves", f"{saves:,.0f}")
    c[1].metric("Engagement", f"{engagement:,.0f}")
    c[2].metric("Follower Growth", f"{growth:,.0f}")
    c[3].metric("Avg Engagement Rate", f"{filtered['engagement_rate'].mean():.2f}%")

    section("KPI Distribution")
    fig = px.bar(kpis, x="KPI", y="Value", title="Selected Period KPI Overview",
                 text_auto=".2s")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    section("Growth Efficiency")
    efficiency = filtered.groupby("topic", as_index=False).agg(
        Reach=("reach","sum"),
        Growth=("follower_growth","sum"),
        Engagement=("engagement","sum")
    )
    efficiency["Growth_per_1000_Reach"] = np.where(
        efficiency["Reach"]>0,
        efficiency["Growth"]/efficiency["Reach"]*1000,0
    )
    bar_chart(efficiency, "topic", "Growth_per_1000_Reach",
              "Follower Growth per 1,000 Reach")

# ============================================================
# 10. DATA EXPLORER
# ============================================================
elif page == "🔎 Data Explorer":
    st.markdown('<div class="big-title">🔎 Data Explorer</div>', unsafe_allow_html=True)
    st.caption("Inspect the structured social media dataset used by the dashboard.")

    search = st.text_input("Search comments, topics, formats or content IDs")
    view = filtered.copy()

    if search:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = view[mask]

    st.write(f"Showing **{len(view)}** records.")

    st.dataframe(view, use_container_width=True, height=620, hide_index=True)

    st.download_button(
        "⬇️ Download Filtered Dataset",
        view.to_csv(index=False),
        "filtered_social_media_dataset.csv",
        "text/csv"
    )

    section("Dataset Statistics")
    st.dataframe(view.describe(include="all").T, use_container_width=True)

# ============================================================
# 11. PROJECT INSIGHTS
# ============================================================
elif page == "📑 Project Insights":
    st.markdown('<div class="big-title">📑 Project Insights & Strategy Report</div>', unsafe_allow_html=True)
    st.caption("Presentation-ready summary of the analytics system.")

    topic_best = filtered.groupby("topic")["engagement"].mean().idxmax()
    topic_viral = filtered.groupby("topic")["viral_score"].mean().idxmax()
    topic_growth = filtered.groupby("topic")["follower_growth"].mean().idxmax()
    format_best = filtered.groupby("format")["engagement"].mean().idxmax()
    hook_best = filtered.groupby("hook")["engagement"].mean().idxmax()

    st.markdown(f"""
    <div class="dashboard-card">
    <h3>🎯 Key Findings</h3>
    <ul>
    <li><b>Highest average engagement topic:</b> {topic_best}</li>
    <li><b>Highest viral-potential topic:</b> {topic_viral}</li>
    <li><b>Highest average follower-growth topic:</b> {topic_growth}</li>
    <li><b>Best-performing format:</b> {format_best}</li>
    <li><b>Best-performing hook:</b> {hook_best}</li>
    <li><b>Average engagement rate:</b> {filtered['engagement_rate'].mean():.2f}%</li>
    <li><b>Average retention:</b> {filtered['retention_rate'].mean():.2f}%</li>
    <li><b>Relatability rate:</b> {(filtered['relatable']=="Relatable").mean()*100:.2f}%</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    section("System Architecture")
    arch = pd.DataFrame({
        "Module": [
            "Data Extraction",
            "Performance Tracking",
            "Virality Prediction",
            "NLP Sentiment",
            "A/B Testing",
            "Optimization",
            "Visualization",
            "Forecasting"
        ],
        "Purpose": [
            "Collect social media performance data",
            "Measure reach and engagement",
            "Estimate organic viral potential",
            "Analyze audience emotion and relatability",
            "Compare content experiments",
            "Recommend stronger content choices",
            "Present KPIs and trends",
            "Project future growth"
        ]
    })
    st.dataframe(arch, use_container_width=True, hide_index=True)

    section("Recommended Weekly Content Strategy")
    strategy = pd.DataFrame({
        "Day": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "Content Focus": [
            "Problem awareness","Relatable story","High-retention short",
            "Educational value","Emotional hook","Community question",
            "Best-performing topic"
        ],
        "Primary KPI": [
            "Comments","Shares","Retention","Saves","Shares","Comments","Follower Growth"
        ]
    })
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.success(
        "This dashboard combines performance analytics, virality scoring, NLP-style "
        "sentiment analysis, A/B testing, recommendation logic and forecasting "
        "into one presentation-ready system."
    )

# -------------------- FOOTER --------------------
st.sidebar.markdown("---")
st.sidebar.caption("Data-Driven Social Engagement Initiative")
st.sidebar.caption("Major Project • Analytics Dashboard")

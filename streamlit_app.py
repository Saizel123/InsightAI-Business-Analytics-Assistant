import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sqlite3
import requests
from io import BytesIO

st.set_page_config(
    page_title="InsightAI - Business Analytics Copilot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 InsightAI - Business Analytics Copilot")
st.write("Upload or explore business data, check data quality, and generate analytics insights.")

st.markdown(
    """
    **InsightAI** is a business analytics copilot that combines dashboarding, 
    data-quality checks, SQL analysis, executive reporting, and simple AI-style business insights.

    Use the sidebar to filter the data by year, region, category, segment, and state.
    """
)

with st.expander("What this app can do"):
    st.markdown(
        """
        - Analyze sales, profit, orders, customers, and profit margin
        - Detect missing values, duplicate rows, and data-quality issues
        - Visualize sales and profitability trends
        - Generate an executive business summary
        - Recommend business actions based on selected data
        - Answer common business questions using the Ask Your Data assistant
        - Generate and run SQL queries using an in-memory SQLite database
        - Download an executive report
        """
    )

DATA_PATH = Path("data/superstore.csv")

BUDGET_PATH = Path("data/budget_actual.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="latin1")

    # Convert date columns
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Handle missing postal codes
    df["Postal Code"] = df["Postal Code"].fillna("Unknown").astype(str)

    # Create useful time columns
    df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Order Year"] = df["Order Date"].dt.year

    return df


@st.cache_data
def load_budget_data(path):
    budget_df = pd.read_csv(path)

    budget_df["Month"] = pd.to_datetime(budget_df["Month"], errors="coerce")
    budget_df["Budget"] = pd.to_numeric(budget_df["Budget"], errors="coerce")
    budget_df["Actual Cost"] = pd.to_numeric(budget_df["Actual Cost"], errors="coerce")
    budget_df["Revenue"] = pd.to_numeric(budget_df["Revenue"], errors="coerce")

    budget_df["Profit"] = budget_df["Revenue"] - budget_df["Actual Cost"]
    budget_df["Variance"] = budget_df["Budget"] - budget_df["Actual Cost"]
    budget_df["Variance %"] = (
        budget_df["Variance"] / budget_df["Budget"] * 100
    ).round(2)

    budget_df["Month Label"] = budget_df["Month"].dt.to_period("M").astype(str)

    return budget_df


def prepare_sql_table(df):
    sql_df = df.copy()

    sql_df.columns = (
        sql_df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    sql_df["Order_Date"] = pd.to_datetime(sql_df["Order_Date"], errors="coerce").astype(str)
    sql_df["Ship_Date"] = pd.to_datetime(sql_df["Ship_Date"], errors="coerce").astype(str)

    conn = sqlite3.connect(":memory:")
    sql_df.to_sql("superstore", conn, index=False, if_exists="replace")

    return conn


def generate_hf_summary(prompt, hf_token):
    api_url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You are a concise business analytics assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 250,
        "temperature": 0.3
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"Hugging Face API error: {response.status_code} - {response.text}"

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]

        return str(result)

    except Exception as e:
        return f"Error while calling Hugging Face API: {e}"

def create_excel_report(sales_summary_df, recommendations_df, filename_sheet_name="Summary"):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sales_summary_df.to_excel(writer, index=False, sheet_name=filename_sheet_name)
        recommendations_df.to_excel(writer, index=False, sheet_name="Recommendations")

    output.seek(0)
    return output

if DATA_PATH.exists():
    st.sidebar.header("📁 Data Source")

    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help="Optional. Paste your Hugging Face token to enable AI-generated summaries."
    )


    st.sidebar.caption("Your token is used only during this session and is not stored in the app.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your own CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Uploaded dataset loaded successfully!")

        # Try to convert common date columns
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

    else:
        df = load_data(DATA_PATH)
        st.success("Sample Superstore dataset loaded successfully!")

    conn = prepare_sql_table(df)
    required_columns = [
        "Order Date",
        "Ship Date",
        "Order ID",
        "Customer ID",
        "Region",
        "Category",
        "Sub-Category",
        "Sales",
        "Quantity",
        "Discount",
        "Profit"
    ]

    missing_required_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_required_columns:
        st.error(
            "The uploaded dataset does not match the expected Superstore format. "
            "Please upload a dataset with the required sales/business columns."
        )

        st.write("Missing required columns:")
        st.write(missing_required_columns)

        st.stop()
    
    with st.sidebar.expander("Expected Dataset Format"):
        st.write("Your uploaded file should contain these columns:")

        expected_schema = pd.DataFrame({
            "Column": required_columns,
            "Purpose": [
                "Order date for time-based analysis",
                "Shipping date",
                "Unique order identifier",
                "Unique customer identifier",
                "Sales region",
                "Product category",
                "Product sub-category",
                "Sales amount",
                "Quantity sold",
                "Discount applied",
                "Profit amount"
            ]
        })

        st.dataframe(expected_schema, use_container_width=True)
    if df.empty:
        st.error("The dataset is empty. Please upload a file with data.")
        st.stop()
    # Sidebar filters
    st.sidebar.header("🔎 Dashboard Filters")

    min_year = int(df["Order Year"].min())
    max_year = int(df["Order Year"].max())

    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    selected_regions = st.sidebar.multiselect(
        "Select Region",
        options=sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique())
    )

    selected_categories = st.sidebar.multiselect(
        "Select Category",
        options=sorted(df["Category"].dropna().unique()),
        default=sorted(df["Category"].dropna().unique())
    )

    selected_segments = st.sidebar.multiselect(
        "Select Segment",
        options=sorted(df["Segment"].dropna().unique()),
        default=sorted(df["Segment"].dropna().unique())
    )

    selected_states = st.sidebar.multiselect(
        "Select State",
        options=sorted(df["State"].dropna().unique()),
        default=sorted(df["State"].dropna().unique())
    )

    filtered_df = df[
        (df["Order Year"] >= selected_years[0]) &
        (df["Order Year"] <= selected_years[1]) &
        (df["Region"].isin(selected_regions)) &
        (df["Category"].isin(selected_categories)) &
        (df["Segment"].isin(selected_segments)) &
        (df["State"].isin(selected_states))
    ]    
    st.sidebar.write(f"Filtered rows: {filtered_df.shape[0]:,}")
    tab_overview, tab_quality, tab_profit, tab_controlling, tab_ask, tab_sql, tab_raw = st.tabs(
        [
            "📌 Business Dashboard",
            "🧹 Data Quality",
            "📊 Profitability Insights",
            "💼 Controlling Analysis",
            "💬 Ask Your Data",
            "🧾 SQL Lab",
            "📄 Raw Data"
        ]
     )

    
    # KPI overview
    with tab_overview:
        st.subheader("📌 Business KPI Overview")

        total_sales = filtered_df["Sales"].sum()
        total_profit = filtered_df["Profit"].sum()
        total_orders = filtered_df["Order ID"].nunique()
        total_customers = filtered_df["Customer ID"].nunique()
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Sales", f"${total_sales:,.2f}")
            st.metric("Total Orders", f"{total_orders:,}")

        with col2:
            st.metric("Total Profit", f"${total_profit:,.2f}")
            st.metric("Total Customers", f"{total_customers:,}")

        with col3:
            st.metric("Average Order Value", f"${avg_order_value:,.2f}")
            st.metric("Profit Margin", f"{profit_margin:.2f}%")
#-------------------------------------------------------------------

#-------------------------------------------------------------------
    # Data quality report
    with tab_quality:
        st.subheader("🧹 Data Quality Report")
        missing_values = filtered_df.isnull().sum()
        missing_percent = (missing_values / len(filtered_df) * 100).round(2) if len(filtered_df) > 0 else 0
        duplicate_rows = filtered_df.duplicated().sum()

        quality_col1, quality_col2, quality_col3 = st.columns(3)

        with quality_col1:
            st.metric("Total Missing Values", int(missing_values.sum()))

        with quality_col2:
            st.metric("Columns with Missing Values", int((missing_values > 0).sum()))

        with quality_col3:
            st.metric("Duplicate Rows", int(duplicate_rows))

        data_quality = pd.DataFrame({
            "Column": filtered_df.columns,
            "Missing Values": missing_values.values,
            "Missing %": missing_percent.values,
            "Data Type": filtered_df.dtypes.astype(str).values,
            "Unique Values": filtered_df.nunique().values
        })

        data_quality = data_quality.sort_values("Missing %", ascending=False)

        st.dataframe(data_quality, use_container_width=True)

        if duplicate_rows > 0:
            st.warning(f"The filtered dataset contains {duplicate_rows} duplicate rows. These may need to be reviewed before analysis.")
        else:
            st.success("No duplicate rows found.")

        high_missing_cols = data_quality[data_quality["Missing %"] > 20]["Column"].tolist()

        if high_missing_cols:
            st.warning(f"Columns with more than 20% missing values: {', '.join(high_missing_cols)}")
        else:
            st.success("No columns have more than 20% missing values.")
    #------------------------------------------------------------------------------
    # Business charts and overview analysis
    with tab_overview:
        st.subheader("📈 Business Performance Charts")

        chart_col1, chart_col2 = st.columns(2)

        yearly_sales = (
            filtered_df.groupby("Order Year", as_index=False)["Sales"]
            .sum()
            .sort_values("Order Year")
        )

        fig_yearly_sales = px.line(
            yearly_sales,
            x="Order Year",
            y="Sales",
            markers=True,
            title="Yearly Sales Trend"
        )

        fig_yearly_sales.update_xaxes(
            tickmode="linear",
            dtick=1
        )

        with chart_col1:
            st.plotly_chart(fig_yearly_sales, use_container_width=True)

        category_profit = (
            filtered_df.groupby("Category", as_index=False)["Profit"]
            .sum()
            .sort_values("Profit", ascending=False)
        )

        fig_category_profit = px.bar(
            category_profit,
            x="Category",
            y="Profit",
            title="Profit by Category",
            text_auto=".2s"
        )

        with chart_col2:
            st.plotly_chart(fig_category_profit, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        region_sales = (
            filtered_df.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )

        fig_region_sales = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            title="Sales by Region",
            text_auto=".2s"
        )

        with chart_col3:
            st.plotly_chart(fig_region_sales, use_container_width=True)

        subcat_sales = (
            filtered_df.groupby("Sub-Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
            .head(10)
        )

        fig_subcat_sales = px.bar(
            subcat_sales,
            x="Sales",
            y="Sub-Category",
            orientation="h",
            title="Top 10 Sub-Categories by Sales",
            text_auto=".2s"
        )

        with chart_col4:
            st.plotly_chart(fig_subcat_sales, use_container_width=True)

        # ------------------------------------------------------------
        st.subheader("🏆 Top Performers")

        top_col1, top_col2 = st.columns(2)

        top_products = (
            filtered_df.groupby("Product Name", as_index=False)
            .agg({
                "Sales": "sum",
                "Profit": "sum",
                "Quantity": "sum"
            })
            .sort_values("Sales", ascending=False)
            .head(10)
        )

        top_products["Sales"] = top_products["Sales"].round(2)
        top_products["Profit"] = top_products["Profit"].round(2)

        with top_col1:
            st.markdown("#### Top 10 Products by Sales")
            st.dataframe(top_products, use_container_width=True)

        top_customers = (
            filtered_df.groupby("Customer Name", as_index=False)
            .agg({
                "Sales": "sum",
                "Profit": "sum",
                "Order ID": "nunique"
            })
            .rename(columns={"Order ID": "Total Orders"})
            .sort_values("Sales", ascending=False)
            .head(10)
        )

        top_customers["Sales"] = top_customers["Sales"].round(2)
        top_customers["Profit"] = top_customers["Profit"].round(2)

        with top_col2:
            st.markdown("#### Top 10 Customers by Sales")
            st.dataframe(top_customers, use_container_width=True)

        # ------------------------------------------------------------
        st.subheader("⚠️ Loss & Risk Analysis")

        risk_col1, risk_col2 = st.columns(2)

        loss_products = (
            filtered_df.groupby("Product Name", as_index=False)
            .agg({
                "Sales": "sum",
                "Profit": "sum",
                "Discount": "mean",
                "Quantity": "sum"
            })
            .sort_values("Profit", ascending=True)
            .head(10)
        )

        loss_products = loss_products[loss_products["Profit"] < 0]

        loss_products["Sales"] = loss_products["Sales"].round(2)
        loss_products["Profit"] = loss_products["Profit"].round(2)
        loss_products["Discount"] = loss_products["Discount"].round(3)

        with risk_col1:
            st.markdown("#### Top Loss-Making Products")
            if not loss_products.empty:
                st.dataframe(loss_products, use_container_width=True)
            else:
                st.success("No loss-making products found for the selected filters.")

        loss_regions = (
            filtered_df.groupby("Region", as_index=False)
            .agg({
                "Sales": "sum",
                "Profit": "sum",
                "Discount": "mean"
            })
            .sort_values("Profit", ascending=True)
        )

        loss_regions = loss_regions[loss_regions["Profit"] < 0]

        loss_regions["Sales"] = loss_regions["Sales"].round(2)
        loss_regions["Profit"] = loss_regions["Profit"].round(2)
        loss_regions["Discount"] = loss_regions["Discount"].round(3)

        with risk_col2:
            st.markdown("#### Loss-Making Regions")
            if not loss_regions.empty:
                st.dataframe(loss_regions, use_container_width=True)
            else:
                st.success("No loss-making regions found for the selected filters.")
#----------------------------------------------------------------

#----------------------------------------------------------------
    with tab_profit:
        st.subheader("📊 Profitability Analysis")

        subcat_profitability = (
            filtered_df.groupby("Sub-Category", as_index=False)
            .agg({
                "Sales": "sum",
                "Profit": "sum",
                "Discount": "mean"
            })
        )

        fig_profitability = px.scatter(
            subcat_profitability,
            x="Sales",
            y="Profit",
            size="Discount",
            hover_name="Sub-Category",
            title="Sales vs Profit by Sub-Category",
            labels={
                "Sales": "Total Sales",
                "Profit": "Total Profit",
                "Discount": "Average Discount"
            }
        )

        st.plotly_chart(fig_profitability, use_container_width=True)

        # Profitability interpretation
        high_sales_low_profit = subcat_profitability[
            (subcat_profitability["Sales"] > subcat_profitability["Sales"].median()) &
            (subcat_profitability["Profit"] < subcat_profitability["Profit"].median())
        ].sort_values("Sales", ascending=False)

        loss_making_subcats = subcat_profitability[
            subcat_profitability["Profit"] < 0
        ].sort_values("Profit")

        st.markdown("### Profitability Interpretation")

        highlighted_subcats = set()

        if not loss_making_subcats.empty:
            worst_loss = loss_making_subcats.iloc[0]
            highlighted_subcats.add(worst_loss["Sub-Category"])

            st.warning(
                f"**{worst_loss['Sub-Category']}** is the main profitability concern: "
                f"it generated USD {worst_loss['Sales']:,.2f} in sales but resulted in "
                f"USD {worst_loss['Profit']:,.2f} profit. Review discounts, pricing, or costs for this sub-category."
            )

        high_sales_low_profit = high_sales_low_profit[
            ~high_sales_low_profit["Sub-Category"].isin(highlighted_subcats)
        ]

        if not high_sales_low_profit.empty:
            risky_subcat = high_sales_low_profit.iloc[0]

            st.info(
                f"**{risky_subcat['Sub-Category']}** also deserves attention because it has strong sales "
                f"but comparatively weaker profit contribution. This may be a margin-improvement opportunity."
            )

        if loss_making_subcats.empty and high_sales_low_profit.empty:
            st.success("No major profitability concern detected based on the selected filters.")

#----------------------------------------------------------------

#----------------------------------------------------------------
    # Profitability insights tab
    with tab_profit:
        st.subheader("🤖 AI Executive Summary")

        top_region = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_category = (
            filtered_df.groupby("Category")["Profit"]
            .sum()
            .sort_values(ascending=False)
        )

        worst_category = (
            filtered_df.groupby("Category")["Profit"]
            .sum()
            .sort_values(ascending=True)
        )

        top_subcategory = (
            filtered_df.groupby("Sub-Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        recommendations = []

        if not filtered_df.empty:
            summary = f"""
Based on the selected filters, the dataset contains **{filtered_df.shape[0]:,} rows**.

The business generated **USD {total_sales:,.2f} in total sales** and **USD {total_profit:,.2f} in total profit**, with a profit margin of **{profit_margin:.2f}%**.

The strongest region by sales is **{top_region.index[0]}**, generating **USD {top_region.iloc[0]:,.2f}** in sales.

The most profitable category is **{top_category.index[0]}**, contributing **USD {top_category.iloc[0]:,.2f}** in profit.

The weakest category by profit is **{worst_category.index[0]}**, with **USD {worst_category.iloc[0]:,.2f}** in profit.

The top-selling sub-category is **{top_subcategory.index[0]}**, with **USD {top_subcategory.iloc[0]:,.2f}** in sales.
"""

            st.markdown(summary)

            st.markdown("### Optional AI-Generated Summary")


            ai_prompt = f"""
You are a business analytics assistant. Summarize the following dashboard in clear business language.

Sales Dashboard:
Rows analyzed: {filtered_df.shape[0]:,}
Total sales: USD {total_sales:,.2f}
Total profit: USD {total_profit:,.2f}
Total orders: {total_orders:,}
Total customers: {total_customers:,}
Average order value: USD {avg_order_value:,.2f}
Profit margin: {profit_margin:.2f}%

Strongest region by sales: {top_region.index[0]} with USD {top_region.iloc[0]:,.2f}
Most profitable category: {top_category.index[0]} with USD {top_category.iloc[0]:,.2f}
Weakest category by profit: {worst_category.index[0]} with USD {worst_category.iloc[0]:,.2f}
Top-selling sub-category: {top_subcategory.index[0]} with USD {top_subcategory.iloc[0]:,.2f}

Controlling / Budget vs Actual Context:
The app also includes a synthetic Budget vs Actual dataset for department-level variance analysis.
It calculates budget, actual cost, revenue, profit, variance, and variance percentage by department.

Give:
1. Three concise business insights.
2. Three recommended business or controlling actions.
3. One sentence explaining why variance analysis is useful for management reporting.
"""



            if hf_token:
                if st.button("Generate AI Summary"):
                    with st.spinner("Generating AI summary with Hugging Face..."):
                        ai_summary = generate_hf_summary(ai_prompt, hf_token)

                    st.write(ai_summary)
                    st.caption("Generated using Hugging Face Inference API.")
            else:
                st.info("Add a Hugging Face API token in the sidebar to enable AI-generated summaries.")

            st.markdown("### Recommended Business Actions")

            if profit_margin < 10:
                recommendations.append(
                    "Review discounting strategy and identify products or regions with low profit margins."
                )
            else:
                recommendations.append(
                    "Maintain current profitability strategy, but continue monitoring discount-heavy categories."
                )

            if worst_category.iloc[0] < 0:
                recommendations.append(
                    f"Investigate the **{worst_category.index[0]}** category because it is currently loss-making."
                )
            else:
                recommendations.append(
                    f"Review the **{worst_category.index[0]}** category because it has the weakest profit contribution."
                )

            if top_region.iloc[0] > 0:
                recommendations.append(
                    f"Use the **{top_region.index[0]}** region as a benchmark to understand what drives stronger sales performance."
                )

            recommendations.append(
                f"Prioritize high-performing sub-categories such as **{top_subcategory.index[0]}**, while checking whether sales growth is also profitable."
            )

            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")

            if profit_margin < 10:
                st.warning(
                    "Profit margin is relatively low. The business should review discounting, loss-making products, and regional profitability."
                )
            else:
                st.success("Profit margin looks healthy based on the selected data.")

            st.subheader("📄 Download Executive Report")

            report_text = f"""
InsightAI - Business Analytics Executive Report

Selected Data Overview
----------------------
Rows analyzed: {filtered_df.shape[0]:,}
Total sales: USD {total_sales:,.2f}
Total profit: USD {total_profit:,.2f}
Total orders: {total_orders:,}
Total customers: {total_customers:,}
Average order value: USD {avg_order_value:,.2f}
Profit margin: {profit_margin:.2f}%

Key Insights
------------
Strongest region by sales: {top_region.index[0]} with USD {top_region.iloc[0]:,.2f}
Most profitable category: {top_category.index[0]} with USD {top_category.iloc[0]:,.2f}
Weakest category by profit: {worst_category.index[0]} with USD {worst_category.iloc[0]:,.2f}
Top-selling sub-category: {top_subcategory.index[0]} with USD {top_subcategory.iloc[0]:,.2f}

Recommended Actions
-------------------
"""

            for i, rec in enumerate(recommendations, start=1):
                clean_rec = rec.replace("**", "")
                report_text += f"{i}. {clean_rec}\n"
 
            report_text += "\nNote: This report was automatically generated from the filtered dashboard data and should be reviewed by a business analyst before decision-making.\n"


            st.download_button(
                label="Download Executive Report",
                data=report_text,
                file_name="insightai_executive_report.txt",
                mime="text/plain"
            )

            sales_summary_df = pd.DataFrame({
                "Metric": [
                    "Rows analyzed",
                    "Total sales",
                    "Total profit",
                    "Total orders",
                    "Total customers",
                    "Average order value",
                    "Profit margin",
                    "Strongest region by sales",
                    "Most profitable category",
                    "Weakest category by profit",
                    "Top-selling sub-category"
                ],
                "Value": [
                    f"{filtered_df.shape[0]:,}",
                    f"USD {total_sales:,.2f}",
                    f"USD {total_profit:,.2f}",
                    f"{total_orders:,}",
                    f"{total_customers:,}",
                    f"USD {avg_order_value:,.2f}",
                    f"{profit_margin:.2f}%",
                    top_region.index[0],
                    top_category.index[0],
                    worst_category.index[0],
                    top_subcategory.index[0]
                ]
            })

            recommendations_df = pd.DataFrame({
                "Recommended Management Action": [
                    rec.replace("**", "") for rec in recommendations
                ]
            })

            business_excel = create_excel_report(
                sales_summary_df,
                recommendations_df,
                filename_sheet_name="Business Summary"
            )

            st.download_button(
                label="Download Business Report as Excel",
                data=business_excel,
                file_name="insightai_business_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            

        else:
            st.warning("No data available for the selected filters.")
#----------------------------------------------------------------

#----------------------------------------------------------------
    # Controlling Analysis
    with tab_controlling:
        st.subheader("💼 Budget vs Actual / Variance Analysis")

        if BUDGET_PATH.exists():
            budget_df = load_budget_data(BUDGET_PATH)

            st.info(
                "This section uses a synthetic Budget vs Actual dataset to demonstrate "
                "controlling-style variance analysis. It is independent from the Superstore sales filters."
            )

            ctrl_filter_col1, ctrl_filter_col2 = st.columns(2)

            with ctrl_filter_col1:
                selected_departments = st.multiselect(
                    "Select Department",
                    options=sorted(budget_df["Department"].dropna().unique()),
                    default=sorted(budget_df["Department"].dropna().unique())
                )

            with ctrl_filter_col2:
                selected_months = st.multiselect(
                    "Select Month",
                    options=sorted(budget_df["Month Label"].dropna().unique()),
                    default=sorted(budget_df["Month Label"].dropna().unique())
                )

            budget_filtered = budget_df[
                (budget_df["Department"].isin(selected_departments)) &
                (budget_df["Month Label"].isin(selected_months))
            ]

            if budget_filtered.empty:
                st.warning("No budget data available for the selected filters.")
                st.stop()

            total_budget = budget_filtered["Budget"].sum()
            total_actual = budget_filtered["Actual Cost"].sum()
            total_revenue = budget_filtered["Revenue"].sum()
            total_profit_budget = budget_filtered["Profit"].sum()
            total_variance = budget_filtered["Variance"].sum()
            variance_percent = (total_variance / total_budget * 100) if total_budget > 0 else 0

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

            with ctrl_col1:
                st.metric("Total Budget", f"USD {total_budget:,.2f}")
                st.metric("Total Revenue", f"USD {total_revenue:,.2f}")

            with ctrl_col2:
                st.metric("Actual Cost", f"USD {total_actual:,.2f}")
                st.metric("Profit", f"USD {total_profit_budget:,.2f}")

            with ctrl_col3:
                st.metric("Variance", f"USD {total_variance:,.2f}")
                st.metric("Variance %", f"{variance_percent:.2f}%")

            department_summary = (
                budget_filtered.groupby("Department", as_index=False)
                .agg({
                    "Budget": "sum",
                    "Actual Cost": "sum",
                    "Revenue": "sum",
                    "Profit": "sum",
                    "Variance": "sum"
                })
            )

            department_summary["Variance %"] = (
                department_summary["Variance"] / department_summary["Budget"] * 100
            ).round(2)

            st.markdown("### Department-Level Variance Summary")
            st.dataframe(department_summary, use_container_width=True)

            fig_variance = px.bar(
                department_summary,
                x="Department",
                y="Variance",
                title="Budget Variance by Department",
                text_auto=".2s"
            )

            st.plotly_chart(fig_variance, use_container_width=True)

            fig_budget_actual = px.bar(
                department_summary,
                x="Department",
                y=["Budget", "Actual Cost"],
                barmode="group",
                title="Budget vs Actual Cost by Department"
            )

            st.plotly_chart(fig_budget_actual, use_container_width=True)

            over_budget = department_summary[
                department_summary["Variance"] < 0
            ].sort_values("Variance")

            st.markdown("### Controlling Interpretation")

            if not over_budget.empty:
                worst_department = over_budget.iloc[0]
                st.warning(
                    f"**{worst_department['Department']}** is the largest over-budget department, "
                    f"with a variance of USD {worst_department['Variance']:,.2f} "
                    f"({worst_department['Variance %']:.2f}%). This department should be reviewed for cost control."
                )
            else:
                st.success("No department is over budget based on the selected budget dataset.")

            st.markdown("### Controlling Summary")

            best_department = department_summary.sort_values("Variance", ascending=False).iloc[0]
            highest_profit_department = department_summary.sort_values("Profit", ascending=False).iloc[0]

            controlling_summary = f"""
The total budget was **USD {total_budget:,.2f}**, while actual cost was **USD {total_actual:,.2f}**, resulting in an overall variance of **USD {total_variance:,.2f}**.

The largest over-budget department is **{worst_department['Department'] if not over_budget.empty else 'None'}**.

The department closest to budget is **{best_department['Department']}**, with a variance of **USD {best_department['Variance']:,.2f}**.

The highest-profit department is **{highest_profit_department['Department']}**, generating **USD {highest_profit_department['Profit']:,.2f}** in profit.
"""

            st.markdown(controlling_summary)

            st.markdown("### Suggested Controlling Actions")

            st.write("1. Review departments with negative variance to identify overspending drivers.")
            st.write("2. Compare high-cost departments against revenue and profit contribution.")
            st.write("3. Use budget variance trends to support monthly controlling and planning discussions.")


            controlling_report_text = f"""
InsightAI - Stakeholder-Ready Controlling Report

Budget vs Actual Overview
-------------------------
Total budget: USD {total_budget:,.2f}
Actual cost: USD {total_actual:,.2f}
Total revenue: USD {total_revenue:,.2f}
Profit: USD {total_profit_budget:,.2f}
Variance: USD {total_variance:,.2f}
Variance %: {variance_percent:.2f}%

Key Controlling Insights
------------------------
Largest over-budget department: {worst_department['Department'] if not over_budget.empty else 'None'}
Department closest to budget: {best_department['Department']}
Highest-profit department: {highest_profit_department['Department']}

Recommended Controlling Actions
--------------------------------
1. Review departments with negative variance to identify overspending drivers.
2. Compare high-cost departments against revenue and profit contribution.
3. Use budget variance trends to support monthly controlling and planning discussions.

Note: This stakeholder-ready controlling report was automatically generated from the Budget vs Actual dashboard. The dataset is synthetic and created for demonstrating controlling and variance analysis workflows.

"""

            st.download_button(
                label="Download Controlling Report",
                data=controlling_report_text,
                file_name="insightai_controlling_report.txt",
                mime="text/plain"
            )


            controlling_summary_df = pd.DataFrame({
                "Metric": [
                    "Total budget",
                    "Actual cost",
                    "Total revenue",
                    "Profit",
                    "Variance",
                    "Variance %",
                    "Largest over-budget department",
                    "Department closest to budget",
                    "Highest-profit department"
                ],
                "Value": [
                    f"USD {total_budget:,.2f}",
                    f"USD {total_actual:,.2f}",
                    f"USD {total_revenue:,.2f}",
                    f"USD {total_profit_budget:,.2f}",
                    f"USD {total_variance:,.2f}",
                    f"{variance_percent:.2f}%",
                    worst_department["Department"] if not over_budget.empty else "None",
                    best_department["Department"],
                    highest_profit_department["Department"]
                ]
            })

            controlling_actions_df = pd.DataFrame({
                "Recommended Controlling Action": [
                    "Review departments with negative variance to identify overspending drivers.",
                    "Compare high-cost departments against revenue and profit contribution.",
                    "Use budget variance trends to support monthly controlling and planning discussions."
                ]
            })

            controlling_excel = BytesIO()

            with pd.ExcelWriter(controlling_excel, engine="openpyxl") as writer:
                controlling_summary_df.to_excel(writer, index=False, sheet_name="Controlling Summary")
                department_summary.to_excel(writer, index=False, sheet_name="Department Variance")
                controlling_actions_df.to_excel(writer, index=False, sheet_name="Recommended Actions")

            controlling_excel.seek(0)

            st.download_button(
                label="Download Controlling Report as Excel",
                data=controlling_excel,
                file_name="insightai_controlling_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.download_button(
                label="Download Budget Variance Summary",
                data=department_summary.to_csv(index=False),
                file_name="budget_variance_summary.csv",
                mime="text/csv"
            )

        else:
            st.error("Could not find data/budget_actual.csv. Please check that the file exists.")

#----------------------------------------------------------------


#----------------------------------------------------------------
    # Ask Your Data
    with tab_ask:
        st.subheader("💬 Ask Your Data")

        with st.expander("Example questions you can ask"):
            st.markdown(
                """
                - What is the total sales?
                - What is the total profit?
                - What is the profit margin?
                - Which region has the highest sales?
                - Which region has the lowest sales?
                - Which category is most profitable?
                - Which category is least profitable?
                - What is the top product?
                - Which product is loss making?
                - Who is the top customer?
                - Which sub-category has the highest discount?
                """
            )

        user_question = st.text_input(
            "Ask a business question about the filtered dataset:",
            placeholder="Example: Which region has the highest sales?"
        )

        if user_question:
            question = user_question.lower()

            if "highest sales" in question or "top region" in question or "best region" in question:
                region_sales_answer = (
                    filtered_df.groupby("Region")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not region_sales_answer.empty:
                    st.info(
                        f"The region with the highest sales is **{region_sales_answer.index[0]}** "
                        f"with **USD {region_sales_answer.iloc[0]:,.2f}** in sales."
                    )

            elif "lowest sales" in question or "worst region" in question:
                region_sales_answer = (
                    filtered_df.groupby("Region")["Sales"]
                    .sum()
                    .sort_values(ascending=True)
                )

                if not region_sales_answer.empty:
                    st.info(
                        f"The region with the lowest sales is **{region_sales_answer.index[0]}** "
                        f"with **USD {region_sales_answer.iloc[0]:,.2f}** in sales."
                    )

            elif "highest profit" in question or "most profitable" in question:
                category_profit_answer = (
                    filtered_df.groupby("Category")["Profit"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not category_profit_answer.empty:
                    st.info(
                        f"The most profitable category is **{category_profit_answer.index[0]}** "
                        f"with **USD {category_profit_answer.iloc[0]:,.2f}** in profit."
                    )

            elif "lowest profit category" in question or "least profitable category" in question:
                category_profit_answer = (
                    filtered_df.groupby("Category")["Profit"]
                    .sum()
                    .sort_values(ascending=True)
                )

                if not category_profit_answer.empty:
                    st.info(
                        f"The weakest category by profit is **{category_profit_answer.index[0]}** "
                        f"with **USD {category_profit_answer.iloc[0]:,.2f}** in profit."
                    )

            elif "total sales" in question:
                st.info(f"Total sales for the selected data are **USD {total_sales:,.2f}**.")

            elif "total profit" in question:
                st.info(f"Total profit for the selected data is **USD {total_profit:,.2f}**.")

            elif "profit margin" in question:
                st.info(f"The profit margin for the selected data is **{profit_margin:.2f}%**.")

            elif "orders" in question:
                st.info(f"The selected data contains **{total_orders:,} unique orders**.")

            elif "customers" in question:
                st.info(f"The selected data contains **{total_customers:,} unique customers**.")

            elif "top product" in question or "best product" in question:
                product_sales_answer = (
                    filtered_df.groupby("Product Name")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not product_sales_answer.empty:
                    st.info(
                        f"The top product by sales is **{product_sales_answer.index[0]}** "
                        f"with **USD {product_sales_answer.iloc[0]:,.2f}** in sales."
                    )

            elif (
                "worst product" in question
                or "loss product" in question
                or "loss making product" in question
                or "product is loss" in question
                or "loss-making product" in question
            ):
                product_profit_answer = (
                    filtered_df.groupby("Product Name")["Profit"]
                    .sum()
                    .sort_values(ascending=True)
                )

                if not product_profit_answer.empty:
                    st.info(
                        f"The weakest product by profit is **{product_profit_answer.index[0]}** "
                        f"with **USD {product_profit_answer.iloc[0]:,.2f}** in profit."
                    )

            elif "top customer" in question or "best customer" in question:
                customer_sales_answer = (
                    filtered_df.groupby("Customer Name")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not customer_sales_answer.empty:
                    st.info(
                        f"The top customer by sales is **{customer_sales_answer.index[0]}** "
                        f"with **USD {customer_sales_answer.iloc[0]:,.2f}** in sales."
                    )

            elif "highest discount" in question or "discount" in question:
                discount_answer = (
                    filtered_df.groupby("Sub-Category")["Discount"]
                    .mean()
                    .sort_values(ascending=False)
                )

                if not discount_answer.empty:
                    st.info(
                        f"The sub-category with the highest average discount is **{discount_answer.index[0]}** "
                        f"with an average discount of **{discount_answer.iloc[0]:.2f}**."
                    )

            elif "top category" in question or "best category" in question:
                category_sales_answer = (
                    filtered_df.groupby("Category")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                )

                if not category_sales_answer.empty:
                    st.info(
                        f"The top category by sales is **{category_sales_answer.index[0]}** "
                        f"with **USD {category_sales_answer.iloc[0]:,.2f}** in sales."
                    )

            else:
                st.warning(
                    "I can currently answer questions about total sales, total profit, profit margin, "
                    "orders, customers, best/worst region, top category, top product, worst product, "
                    "top customer, highest discount, and most/least profitable category."
                )
#----------------------------------------------------------------

#----------------------------------------------------------------
    # SQL Query Generator
    with tab_sql:
        st.subheader("🧾 SQL Query Generator")

        sql_question = st.selectbox(
            "Choose a business question to generate and run SQL:",
            [
                "Select a question",
                "Total sales by region",
                "Total profit by category",
                "Monthly sales trend",
                "Top 10 products by sales",
                "Top 10 customers by sales",
                "Average discount by category",
                "Number of orders by segment"
            ]
        )

        sql_queries = {
            "Total sales by region": """
SELECT 
    Region,
    ROUND(SUM(Sales), 2) AS total_sales
FROM superstore
GROUP BY Region
ORDER BY total_sales DESC;
""",
            "Total profit by category": """
SELECT 
    Category,
    ROUND(SUM(Profit), 2) AS total_profit
FROM superstore
GROUP BY Category
ORDER BY total_profit DESC;
""",
            "Monthly sales trend": """
SELECT 
    substr(Order_Date, 1, 7) AS order_month,
    ROUND(SUM(Sales), 2) AS monthly_sales
FROM superstore
GROUP BY order_month
ORDER BY order_month;
""",
            "Top 10 products by sales": """
SELECT 
    Product_Name,
    ROUND(SUM(Sales), 2) AS total_sales
FROM superstore
GROUP BY Product_Name
ORDER BY total_sales DESC
LIMIT 10;
""",
            "Top 10 customers by sales": """
SELECT 
    Customer_Name,
    ROUND(SUM(Sales), 2) AS total_sales
FROM superstore
GROUP BY Customer_Name
ORDER BY total_sales DESC
LIMIT 10;
""",
            "Average discount by category": """
SELECT 
    Category,
    ROUND(AVG(Discount), 3) AS average_discount
FROM superstore
GROUP BY Category
ORDER BY average_discount DESC;
""",
            "Number of orders by segment": """
SELECT 
    Segment,
    COUNT(DISTINCT Order_ID) AS total_orders
FROM superstore
GROUP BY Segment
ORDER BY total_orders DESC;
"""
        }

        if sql_question != "Select a question":
            selected_query = sql_queries[sql_question]

            st.markdown("#### Generated SQL")
            st.code(selected_query, language="sql")

            try:
                query_result = pd.read_sql_query(selected_query, conn)

                st.markdown("#### Query Result")
                st.dataframe(query_result, use_container_width=True)

            except Exception as e:
                st.error(f"SQL query failed: {e}")

        st.markdown("### Custom SQL Playground")

        st.write(
            "Write your own SQL query using the `superstore` table. "
            "Use underscore-based column names like `Order_ID`, `Product_Name`, `Sales`, `Profit`, `Region`, and `Category`."
        )

        custom_query = st.text_area(
            "Enter SQL query:",
            value="""
SELECT 
    Region,
    Category,
    ROUND(SUM(Sales), 2) AS total_sales,
    ROUND(SUM(Profit), 2) AS total_profit
FROM superstore
GROUP BY Region, Category
ORDER BY total_sales DESC
LIMIT 10;
""",
            height=220
        )

        if st.button("Run Custom SQL Query"):
            try:
                custom_result = pd.read_sql_query(custom_query, conn)
                st.dataframe(custom_result, use_container_width=True)
            except Exception as e:
                st.error(f"Custom SQL query failed: {e}")



#----------------------------------------------------------------

#----------------------------------------------------------------
    # Raw data tab
    with tab_raw:
        st.subheader("📄 Raw Data")

        with st.expander("View Dataset Preview", expanded=True):
            st.dataframe(filtered_df.head(20), use_container_width=True)

        with st.expander("View Dataset Overview"):
            overview_col1, overview_col2, overview_col3 = st.columns(3)

            with overview_col1:
                st.metric("Rows", filtered_df.shape[0])

            with overview_col2:
                st.metric("Columns", filtered_df.shape[1])

            with overview_col3:
                st.metric("Duplicate Rows", filtered_df.duplicated().sum())

        st.download_button(
            label="Download Filtered Dataset as CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_superstore_data.csv",
            mime="text/csv"
        )
else:
    st.error("Could not find data/superstore.csv. Please check that the file is in the correct folder.")
import streamlit as st
import pandas as pd
import plotly.express as px

def deep_dive_discount_performance_insights(df_filtered):
    st.subheader("🔎 Deep Dive: Discount Performance Insights")
    try:
        # Fix: Convert object-type columns to string for Streamlit compatibility
        def safe_for_streamlit(df):
            for col in df.columns:
                if df[col].dtype == 'O':
                    df[col] = df[col].astype(str)
            return df
        # Discount performance by Channel (visual)
        if 'Channel' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            channel_discount = df_filtered.groupby('Channel').agg(
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Orders=('OrderID', 'count'),
                Total_Sales=('net_sale', 'sum')
            ).sort_values('Total_Discount', ascending=False)
            channel_discount['Discount_Rate_%'] = (channel_discount['Total_Discount'] / channel_discount['Total_Sales'] * 100).round(2)
            st.markdown("**Top Channels by Total Discount Given:**")
            st.dataframe(safe_for_streamlit(channel_discount.head(10)), use_container_width=True)
            # Bar chart
            fig = px.bar(channel_discount.head(10).reset_index(), x='Channel', y='Total_Discount',
                         title='Top Channels by Total Discount', color='Discount_Rate_%',
                         color_continuous_scale='Reds', text='Orders')
            st.plotly_chart(fig, use_container_width=True)

        # Discount performance by Date (visual)
        if 'Date' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            date_discount = df_filtered.groupby('Date').agg(
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Orders=('OrderID', 'count'),
                Total_Sales=('net_sale', 'sum')
            ).sort_values('Date')
            date_discount['Discount_Rate_%'] = (date_discount['Total_Discount'] / date_discount['Total_Sales'] * 100).round(2)
            st.markdown("**Discount Trends Over Time:**")
            st.dataframe(safe_for_streamlit(date_discount.tail(14)), use_container_width=True)
            # Line chart
            fig = px.line(date_discount.reset_index(), x='Date', y='Discount_Rate_%',
                          title='Discount Rate % Over Time', markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Discount performance by Address (Location) (visual)
        if 'Address' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            address_discount = df_filtered.groupby('Address').agg(
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Orders=('OrderID', 'count'),
                Total_Sales=('net_sale', 'sum')
            ).sort_values('Total_Discount', ascending=False)
            address_discount['Discount_Rate_%'] = (address_discount['Total_Discount'] / address_discount['Total_Sales'] * 100).round(2)
            st.markdown("**Top Locations by Discount Given:**")
            st.dataframe(safe_for_streamlit(address_discount.head(10)), use_container_width=True)
            # Bar chart
            fig = px.bar(address_discount.head(10).reset_index(), x='Address', y='Total_Discount',
                         title='Top Locations by Total Discount', color='Discount_Rate_%',
                         color_continuous_scale='Blues', text='Orders')
            st.plotly_chart(fig, use_container_width=True)

        # Discount performance by Brand (if available)
        if 'Brand' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            brand_discount = df_filtered.groupby('Brand').agg(
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Orders=('OrderID', 'count'),
                Total_Sales=('net_sale', 'sum')
            ).sort_values('Total_Discount', ascending=False)
            brand_discount['Discount_Rate_%'] = (brand_discount['Total_Discount'] / brand_discount['Total_Sales'] * 100).round(2)
            st.markdown("**Top Brands by Discount Given:**")
            st.dataframe(safe_for_streamlit(brand_discount.head(10)), use_container_width=True)
            fig = px.bar(brand_discount.head(10).reset_index(), x='Brand', y='Total_Discount', color='Discount_Rate_%',
                         title='Top Brands by Total Discount', text='Orders', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)

        # Discount performance by Brand+Channel (if both available)
        if all(col in df_filtered.columns for col in ['Brand', 'Channel', 'Calculated_Discount']):
            bc_discount = df_filtered.groupby(['Brand', 'Channel']).agg(
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Orders=('OrderID', 'count')
            ).sort_values('Total_Discount', ascending=False).reset_index()
            st.markdown("**Brand × Channel Discount Matrix (Top 20):**")
            st.dataframe(safe_for_streamlit(bc_discount.head(20)), use_container_width=True)
            # Visual: Heatmap for Brand-Channel discount
            if len(bc_discount) > 0:
                fig = px.density_heatmap(bc_discount, x='Brand', y='Channel', z='Total_Discount',
                                        color_continuous_scale='Viridis', title='Brand × Channel Discount Heatmap')
                st.plotly_chart(fig, use_container_width=True)

        # Deepest single-order discounts (visual)
        if 'Calculated_Discount' in df_filtered.columns:
            deepest_discounts = df_filtered.sort_values('Calculated_Discount', ascending=False).head(10)
            st.markdown("**Orders with the Deepest Discounts:**")
            show_cols = [c for c in ['OrderID','CustomerName','Channel','Address','Date','GrossPrice','Calculated_Discount','DiscountCode'] if c in deepest_discounts.columns]
            st.dataframe(safe_for_streamlit(deepest_discounts[show_cols]), use_container_width=True)
            # Bar chart
            fig = px.bar(deepest_discounts, x='OrderID', y='Calculated_Discount',
                         title='Top 10 Deepest Discount Orders', color='Channel' if 'Channel' in deepest_discounts.columns else None)
            st.plotly_chart(fig, use_container_width=True)

        # Discount code effectiveness (visual)
        if 'DiscountCode' in df_filtered.columns:
            code_effect = df_filtered.groupby('DiscountCode').agg(
                Uses=('OrderID', 'count'),
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean')
            ).sort_values('Total_Discount', ascending=False)
            st.markdown("**Most Used Discount Codes (Top 10):**")
            st.dataframe(safe_for_streamlit(code_effect.head(10)), use_container_width=True)
            # Bar chart
            fig = px.bar(code_effect.head(10).reset_index(), x='DiscountCode', y='Uses',
                         title='Top 10 Most Used Discount Codes', color='Total_Discount',
                         color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)

        # Discount distribution histogram (visual)
        if 'Calculated_Discount' in df_filtered.columns:
            st.markdown("**Discount Amount Distribution (All Orders):**")
            fig = px.histogram(df_filtered, x='Calculated_Discount', nbins=50, title='Discount Amount Distribution', color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        # Discount range analysis (granular)
        if 'Calculated_Discount' in df_filtered.columns:
            df_filtered['Discount_Range'] = pd.cut(df_filtered['Calculated_Discount'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['1-10', '11-25', '26-50', '51-100', '100+'])
            range_stats = df_filtered.groupby('Discount_Range').agg(
                Orders=('OrderID', 'count'),
                Total_Discount=('Calculated_Discount', 'sum'),
                Avg_Discount=('Calculated_Discount', 'mean'),
                Avg_Discount_Rate=('Discount_Percentage', 'mean') if 'Discount_Percentage' in df_filtered.columns else ('Calculated_Discount', 'mean')
            ).sort_index()
            st.markdown("**Discount Range Breakdown:**")
            st.dataframe(safe_for_streamlit(range_stats), use_container_width=True)
            # Visual: Bar chart for discount range breakdown
            fig = px.bar(range_stats.reset_index(), x='Discount_Range', y='Orders', color='Avg_Discount_Rate',
                         title='Orders by Discount Range', color_continuous_scale='Teal')
            st.plotly_chart(fig, use_container_width=True)

        # Insightful observations
        st.markdown("### 💡 Discount Insights Summary")
        insights = []
        if 'Channel' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            top_channel = channel_discount.index[0]
            top_channel_rate = channel_discount.iloc[0]['Discount_Rate_%']
            insights.append(f"• **{top_channel}** gives the highest total discount, with an average discount rate of {top_channel_rate:.1f}%.")
        if 'Address' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            top_address = address_discount.index[0]
            top_address_rate = address_discount.iloc[0]['Discount_Rate_%']
            insights.append(f"• **{top_address}** location has the highest total discount, with an average discount rate of {top_address_rate:.1f}%.")
        if 'Brand' in df_filtered.columns and 'Calculated_Discount' in df_filtered.columns:
            top_brand = brand_discount.index[0]
            top_brand_rate = brand_discount.iloc[0]['Discount_Rate_%']
            insights.append(f"• **{top_brand}** brand has the highest total discount, with an average discount rate of {top_brand_rate:.1f}%.")
        if 'DiscountCode' in df_filtered.columns:
            top_code = code_effect.index[0]
            top_code_uses = code_effect.iloc[0]['Uses']
            insights.append(f"• Discount code **{top_code}** was used {int(top_code_uses)} times, more than any other code.")
        if 'Calculated_Discount' in df_filtered.columns:
            max_discount = df_filtered['Calculated_Discount'].max()
            insights.append(f"• The single largest discount given on an order was {max_discount:,.2f}.")
        for ins in insights:
            st.write(ins)
    except Exception as e:
        st.warning(f"⚠️ Unable to generate deep dive discount insights: {str(e)}")

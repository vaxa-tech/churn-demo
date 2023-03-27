import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# noinspection PyUnresolvedReferences
import vaxa_plotly_branding

st.set_page_config(layout="wide", page_title='Vaxa churn demo',
                   page_icon='♻️', initial_sidebar_state='expanded',
                   menu_items={'Get Help': 'https://vaxaanalytics.com/contact-us/',
                               'Report a bug': 'https://vaxaanalytics.com/contact-us/',
                               'About': 'https://vaxaanalytics.com/contact-us/'})
start_date = datetime.datetime.now()

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>', unsafe_allow_html=True)

# setup sidebar
st.sidebar.markdown('# Churn simulator')
st.sidebar.markdown("""
This tool shows the complex relationship between acquisitions (top of funnel) and churn (bottom of funnel). 
 
It should help you understand how powerful even small reductions in churn can have on long-term horizons.
""")
mode = st.sidebar.radio('Mode', ['Members', 'Revenue'])

# current state inputs in sidebar
st.sidebar.subheader('Current state')
if mode == 'Members':
    starting_value = st.sidebar.number_input('Starting members', 1000, 1_000_000, 20_000)
else:
    starting_value = st.sidebar.number_input('Starting revenue', 1000, 3_000_000, 500_000, step=10_000)

# base case inputs in sidebar
st.sidebar.subheader('Base case inputs')
monthly_churn_rate = st.sidebar.slider('Churn rate (per month)', min_value=0.0, max_value=.2, value=0.025, step=0.001,
                                       format='%.3f')
if mode == 'Members':
    acquisition_per_month = st.sidebar.slider('Members acquired per month', min_value=0, max_value=10_000, value=1_000,
                                              step=100, format='%.0f')
else:
    acquisition_per_month = st.sidebar.slider('Revenue acquired per month', min_value=0, max_value=100_000,
                                              value=25_000,
                                              step=1_000, format='%.0f')

# forecast inputs in sidebar
st.sidebar.subheader('Forecast')
churn_rate_reduction = st.sidebar.slider('Churn rate reduction', min_value=0.0, max_value=1.0, value=.25, step=0.001,
                                         format='%.3f',
                                         help='What % reduction do you think you can achieve in churn rate?')
lower_churn_rate, upper_churn_rate = monthly_churn_rate * (1 - churn_rate_reduction), monthly_churn_rate * (
        1 + churn_rate_reduction)
num_months = st.sidebar.slider('Number of months', min_value=1, max_value=120, value=36, step=1, format='%.0f')

# create a dataframe with the number of members per month
df = pd.DataFrame({
    'date': [start_date],
    'acquisition': [acquisition_per_month],

    'churns_base': [starting_value * monthly_churn_rate],
    'churns_lower': [starting_value * lower_churn_rate],
    'churns_upper': [starting_value * upper_churn_rate],

    'members_base': [starting_value],
    'members_lower': [starting_value],
    'members_upper': [starting_value],
})

for i in range(num_months):
    last_row = df.iloc[-1]
    new_row = {
        'date': last_row['date'] + datetime.timedelta(days=30),
        'acquisition': acquisition_per_month,
        'churns_base': last_row['members_base'] * monthly_churn_rate,
        'churns_lower': last_row['members_lower'] * lower_churn_rate,
        'churns_upper': last_row['members_upper'] * upper_churn_rate,
        'members_base': last_row['members_base'] + acquisition_per_month - last_row[
            'members_base'] * monthly_churn_rate,
        'members_lower': last_row['members_lower'] + acquisition_per_month - last_row[
            'members_lower'] * lower_churn_rate,
        'members_upper': last_row['members_upper'] + acquisition_per_month - last_row[
            'members_upper'] * upper_churn_rate,
    }
    df = pd.concat([df, pd.DataFrame(new_row, index=[i])])

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['members_base'], name=f'{monthly_churn_rate:.2%} churn rate'))
fig.add_trace(go.Scatter(x=df['date'], y=df['members_upper'], name=f'{upper_churn_rate:.2%} churn rate'))
fig.add_trace(go.Scatter(x=df['date'], y=df['members_lower'], name=f'{lower_churn_rate:.2%} churn rate'))
fig.add_hline(y=starting_value, line_dash="dash", line_color="gray", name='starting members')
fig.update_layout(
    title=f"Number of Members, acquiring {acquisition_per_month:,.0f} per month" if mode == 'Members' else
    f"Revenue, acquiring ${acquisition_per_month:,.0f} per month",
    xaxis_title="Date",
    yaxis_title="Number of members" if mode == 'Members' else "Revenue",
    legend_title="Legend",
    # legend top left
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01
    ),
    # remove margin on rhs
    margin=dict(
        l=0, r=0, t=60, b=80),
    height=500,
)
fig.update_yaxes(range=[0, df['members_lower'].max() * 1.1], tickformat=',.0f' if mode == 'Members' else '$,.0f')
fig.update_traces(
    hovertemplate='%{y:,.0f} members' if mode == 'Members' else '$%{y:,.0f}',
    hoverlabel=dict(
        bgcolor='white',
        font_size=16,
        font_family='Plus Jakarta Sans',
    ),
)

st.plotly_chart(fig, theme=None, config={'displayModeBar': False})

# create a heatmap, showing the number of members at number_of_months
# under various churn rates either side of monthly_churn_rate
# and various acquisition rates either side of acquisition_per_month
# the heatmap is coloured by the number of members at the end of the period
num_cells = 100
churn_rates = [
    monthly_churn_rate * (1 - churn_rate_reduction) + i * monthly_churn_rate * churn_rate_reduction * 2 / (
            num_cells - 1) for
    i in range(num_cells)]
acquisitions = [
    acquisition_per_month * (1 - churn_rate_reduction) + i * acquisition_per_month * churn_rate_reduction * 2 / (
            num_cells - 1) for i in range(num_cells)]

rows = []
for churn_rate in churn_rates:
    for acquisition in acquisitions:
        # simulate the final figure at number_of_months from now
        # with churn_rate and acquisition
        members = starting_value
        for i in range(num_months):
            members = members + acquisition - members * churn_rate
        rows.append([churn_rate, acquisition, members])

df = pd.DataFrame(rows, columns=['churn_rate', 'acquisition', 'value'])
pivot = df.pivot(index='acquisition', columns='churn_rate', values='value')

heatmap = px.imshow(pivot, text_auto=False, aspect='auto', origin='lower', color_continuous_scale='Viridis')
min_churn_rate, max_churn_rate = df['churn_rate'].min(), df['churn_rate'].max()
heatmap.update_xaxes(range=[min_churn_rate, max_churn_rate])

# make hovertools pretty
if mode == 'Members':
    hover_template = '<b>Churn rate (%)</b>: %{x:.2%}<br><b>Acquisition (memb./mo)</b>: %{y:,.0f}<br><b>Members</b>: %{z:,.0f}<extra></extra>'
else:
    hover_template = '<b>Churn rate (%)</b>: %{x:.2%}<br><b>Acquisition ($/mo)</b>: %{y:,.0f}<br><b>Revenue</b>: $%{z:,.0f}<extra></extra>'
heatmap.update_traces(
    hovertemplate=hover_template,
    hoverlabel=dict(
        bgcolor='white',
        font_size=16,
        font_family='Plus Jakarta Sans',
    ),
)

# for each column in pivot, find the index closest to starting_members
# and note the coordinates, then add a line to heatmap to show the path of breakeven
coords = []
for churn_rate in pivot.columns:
    # find the index of the value closest to starting_value
    idx = (pivot[churn_rate] - starting_value).abs().idxmin()
    coords.append([churn_rate, idx])

coord_df = pd.DataFrame(coords, columns=['x', 'y'])
heatmap.add_trace(
    go.Scatter(
        x=coord_df['x'],
        y=coord_df['y'],
        mode='lines',
        # smooth lines
        line=dict(shape='spline', dash='dot'),
        connectgaps=True,
        # disable hovertools
        hoverinfo='none',
        showlegend=False,
    )
)
# text annotation above spline
heatmap.add_annotation(
    x=coord_df['x'].mean(),
    y=coord_df['y'].mean(),
    text=f'Growth above this line;<br>decline below this line',
    align='center',
    showarrow=False,
    font=dict(
        family='Plus Jakarta Sans',
        size=16,
        color='#fbaa36',
    ),
    bgcolor='rgba(255,255,255,0.3)',
)

heatmap.update_layout(
    title=f"Number of Members at {num_months} months, starting with {starting_value:,.0f} members" if mode == 'Members' else f"Revenue at {num_months} months, starting with ${starting_value:,.0f}",
    xaxis_title="Churn rate (% per month)",
    yaxis_title="Acquired members per month" if mode == 'Members' else "Acquired revenue per month",
)
heatmap.add_vline(x=monthly_churn_rate, line_dash="dash", line_color="black", name='monthly churn rate',
                  annotation=dict(text='Base case churn rate', textangle=-90, showarrow=False))
heatmap.add_hline(y=acquisition_per_month, line_dash="dash", line_color="black", name='acquisition per month',
                  annotation=dict(text='Base case acquisition', textangle=0, showarrow=False))
heatmap.update_xaxes(showspikes=True, range=[df['churn_rate'].min(), df['churn_rate'].max()],
                     tickmode='array',
                     tickvals=[i for idx, i in enumerate(churn_rates) if idx % 10 == 0],
                     ticktext=[f'{churn_rate:.2%}' for idx, churn_rate in enumerate(churn_rates) if idx % 10 == 0]
                     )
heatmap.update_yaxes(showspikes=True, range=[df['acquisition'].min(), df['acquisition'].max()],
                     tickformat=',.0f' if mode == 'Members' else '$,.0f')
st.plotly_chart(heatmap, theme=None, config={'displayModeBar': False})

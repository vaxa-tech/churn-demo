import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# noinspection PyUnresolvedReferences
import vaxa_plotly_branding

st.set_page_config(
    layout="wide",
    page_title='Vaxa churn demo',
    page_icon='♻️', initial_sidebar_state='expanded',
    menu_items={'Get Help': 'https://vaxaanalytics.com/contact-us/',
                'Report a bug': 'https://vaxaanalytics.com/contact-us/',
                'About': 'https://vaxaanalytics.com/contact-us/'})

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Darker+Grotesque:wght@500&family=Plus+Jakarta+Sans:ital@0;1&display=swap');

            h1, .h1, h2, .h2, h3, .h3 {
			    font-family: 'Darker Grotesque', sans-serif !important;
			    
			}
    
			body  {
			font-family: 'Plus Jakarta Sans', sans-serif; 
			}
			
			
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

query_param_keys = ['client_name', 'first_name', 'mode', 'starting_value', 'per_month', 'churn_rate']
# assemble query params into url
query_param_str = '?' + '&'.join([x + '=' for x in query_param_keys])
print(query_param_str)

query_params = st.experimental_get_query_params()
param_client_name = query_params.get('client_name', ['your organisation'])[0]
param_client_name_possessive = f"{param_client_name}'s" if param_client_name[-1] != 's' else f"{param_client_name}'"
param_first_name = query_params.get('first_name', ['John'])[0]
param_starting_value = int(query_params.get('starting_value', [500_000])[0])
param_value_per_month = int(query_params.get('per_month', [25_000])[0])
param_churn_rate = float(query_params.get('churn_rate', [0.025])[0])

start_date = datetime.datetime.now()

with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.markdown(f"""
## {param_first_name}, let's talk churn.

We built this interactive tool to help demonstrate - in simple terms - how even small reductions in churn rate can have 
huge impacts to your top-line revenue.  

Sure, acquisitions are important - we don't argue that - but relatively small reductions in churn are almost always
easier (and cheaper!) when compared with finding and selling to cold leads. Filling up your bucket is a tiring exercise
when there's a hole at the bottom. 

## But first, a bit about you..
What does current state look like for {param_client_name}? We've prefilled the below with a Guess.. but we'd rather Know. 


""")

st.caption("""__Privacy note: we take data privacy and security seriously. Information you enter here is not stored, 
accessible or used by Vaxa or any related party in any way, shape or form - except for providing the below 
functionality. You're in safe hands.__""")

col1, col2, col3 = st.columns(3)
# current state inputs in sidebar
with col1:
    st.subheader('Current state')
    starting_value = st.number_input('Starting revenue', 1000, 3_000_000, param_starting_value, step=10_000)

with col2:
    # base case inputs in sidebar
    st.subheader('Base case inputs')
    monthly_churn_rate = st.slider('Churn rate (per month)', min_value=0.0, max_value=.2, value=param_churn_rate,
                                   step=0.001,
                                   format='%.3f')
    acquisition_per_month = st.slider('Revenue acquired per month', min_value=0, max_value=100_000,
                                      value=param_value_per_month,
                                      step=1_000, format='%.0f')

with col3:
    # forecast inputs in sidebar
    st.subheader('Forecast')
    churn_rate_reduction = st.slider('Churn rate reduction', min_value=0.001, max_value=1.0, value=.25, step=0.001,
                                     format='%.3f',
                                     help='What % reduction do you think you can achieve in churn rate?')
    lower_churn_rate, upper_churn_rate = monthly_churn_rate * (1 - churn_rate_reduction), monthly_churn_rate * (
            1 + churn_rate_reduction)
    num_months = st.slider('Number of months', min_value=1, max_value=120, value=36, step=1, format='%.0f')

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
future_state_base = int(df.iloc[len(df) - 1]['members_base'])
# future_state_upper = df.iloc[-1, 'members_upper']
# future_state_lower = df.iloc[-1, 'members_lower']


fig_proj = go.Figure()
fig_proj.add_trace(go.Scatter(x=df['date'], y=df['members_base'], name=f'{monthly_churn_rate:.2%} churn rate'))
fig_proj.add_trace(
    go.Scatter(x=df['date'], y=df['members_upper'], name=f'{upper_churn_rate:.2%} churn rate', line=dict(dash='dot')))
fig_proj.add_trace(
    go.Scatter(x=df['date'], y=df['members_lower'], name=f'{lower_churn_rate:.2%} churn rate', line=dict(dash='dot')))
fig_proj.add_hline(y=starting_value, line_dash="dash", line_color="gray", name='starting members')
fig_proj.update_layout(
    title=f"Monthly revenue (MRR), acquiring ${acquisition_per_month:,.0f} per month",
    xaxis_title="Date",
    yaxis_title="Revenue",
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
fig_proj.update_yaxes(range=[0, df['members_lower'].max() * 1.1], tickformat='$,.0f')
fig_proj.update_traces(
    hovertemplate='$%{y:,.0f}',
    hoverlabel=dict(
        bgcolor='white',
        font_size=16,
        font_family='Plus Jakarta Sans',
    ),
)

# calculate the monthly revenue
monthly_revenue = df.copy()
monthly_revenue['month'] = pd.to_datetime(monthly_revenue['date'].apply(lambda x: x.strftime('%Y-%m-01')))
monthly_revenue = monthly_revenue.sort_values(by='date', ascending=True).drop_duplicates(subset='month')
monthly_revenue['members_base_cumsum'] = monthly_revenue['members_base'].cumsum()
monthly_revenue['members_upper_cumsum'] = monthly_revenue['members_upper'].cumsum()
monthly_revenue['members_lower_cumsum'] = monthly_revenue['members_lower'].cumsum()
monthly_revenue['members_base_cumsum_delta'] = monthly_revenue['members_base_cumsum'] - monthly_revenue[
    'members_base_cumsum']
monthly_revenue['members_upper_cumsum_delta'] = monthly_revenue['members_upper_cumsum'] - monthly_revenue[
    'members_base_cumsum']
monthly_revenue['members_lower_cumsum_delta'] = monthly_revenue['members_lower_cumsum'] - monthly_revenue[
    'members_base_cumsum']

final_upper_cumsum = monthly_revenue.iloc[len(monthly_revenue) - 1]['members_upper_cumsum_delta']
final_lower_cumsum = monthly_revenue.iloc[len(monthly_revenue) - 1]['members_lower_cumsum_delta']
final_date = monthly_revenue.iloc[len(monthly_revenue) - 1]['date']

fig_rev = go.Figure()
fig_rev.add_trace(
    go.Scatter(
        x=monthly_revenue['date'],
        y=monthly_revenue['members_base_cumsum_delta'],
        name=f'{monthly_churn_rate:.2%} churn rate'
    )
)
fig_rev.add_trace(
    go.Scatter(
        x=monthly_revenue['date'],
        y=monthly_revenue['members_upper_cumsum_delta'],
        name=f'{upper_churn_rate:.2%} churn rate'
    )
)
fig_rev.add_trace(
    go.Scatter(
        x=monthly_revenue['date'],
        y=monthly_revenue['members_lower_cumsum_delta'],
        name=f'{lower_churn_rate:.2%} churn rate'
    )
)
fig_rev.add_annotation(
    x=final_date,
    y=final_upper_cumsum,
    text=f'${final_upper_cumsum:,.0f} less in your bank'
)
fig_rev.add_annotation(
    x=final_date,
    y=final_lower_cumsum,
    text=f'${final_lower_cumsum:,.0f} more in your bank'
)
fig_rev.add_annotation(
    x=final_date,
    y=0,
    text=f'No change from current state'
)
fig_rev.update_layout(
    title=f"Cumulative revenue, acquiring ${acquisition_per_month:,.0f} per month",
    xaxis_title="Date",
    yaxis_title="Cumulative revenue",
    legend_title="Legend",
    # legend top left
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01
    ),
    # remove margin on rhs
    # margin=dict(
    #     l=0, r=0, t=60, b=80),
    height=750,
)

# create a heatmap, showing the number of members at number_of_months
# under various churn rates either side of monthly_churn_rate
# and various acquisition rates either side of acquisition_per_month
# the heatmap is coloured by the number of members at the end of the period
num_cells = 200
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
heatmap.update_traces(
    hovertemplate='<b>Churn rate (%)</b>: %{x:.2%}<br><b>Acquisition ($/mo)</b>: %{y:,.0f}<br><b>Revenue</b>: $%{z:,.0f}<extra></extra>',
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
    title=f"Revenue at {num_months} months, starting with ${starting_value:,.0f}",
    xaxis_title="Churn rate (% per month)",
    yaxis_title="Acquired revenue per month",
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
                     tickformat='$,.0f')
acq_min, acq_max = df['acquisition'].min(), df['acquisition'].max()
churn_min, churn_max = df['churn_rate'].min(), df['churn_rate'].max()


def place_outcome(churn_rate, acquisition_per_month, on_fig, pivot_df):
    df = pivot_df.reset_index().copy()
    df['acquisition_delta'] = df['acquisition'].sub(acquisition_per_month).abs()
    # find df['acquisition_delta'] closest to zero
    df = df.sort_values(by='acquisition_delta', ascending=True).iloc[0].reset_index().iloc[1:-1]
    df.columns = ['churn_rate', 'revenue']
    df['churn_delta'] = df['churn_rate'].sub(churn_rate).abs()
    df = df.sort_values(by='churn_delta', ascending=True).iloc[0].reset_index()
    df.columns = ['key', 'value']
    revenue = df.loc[df['key'] == 'revenue', 'value'].iloc[0]

    on_fig.add_annotation(x=churn_rate, y=acquisition_per_month, text=f'${revenue:,.0f}/month', font_color='#ffffff',
                          font_family='Plus Jakarta Sans')


place_outcome(
    churn_rate=(churn_min + monthly_churn_rate) / 2,
    acquisition_per_month=(acq_max + acquisition_per_month) / 2,
    on_fig=heatmap, pivot_df=pivot)
place_outcome(
    churn_rate=(churn_min + monthly_churn_rate) / 2 ,
    acquisition_per_month =(acq_min + acquisition_per_month ) / 2,
    on_fig=heatmap, pivot_df=pivot)
place_outcome(
    churn_rate=(churn_max + monthly_churn_rate) / 2,
    acquisition_per_month = (acq_max + acquisition_per_month ) / 2,
    on_fig=heatmap, pivot_df=pivot)
place_outcome(
    churn_rate=(churn_max + monthly_churn_rate) / 2 ,
    acquisition_per_month = (acq_min + acquisition_per_month ) / 2,
    on_fig=heatmap, pivot_df=pivot)

st.header(f'Great! So where is {param_client_name if param_client_name is not None else "you"} currently headed?')
st.markdown(f'''
You've asked us to look at things over a {num_months} timeframe, and we'll first start by projecting the future
based on your current state. 

Below, you'll see the expected revenue per month, 
under your current :blue[{monthly_churn_rate:.2%} churn rate]. We've also added lines for a :violet[{churn_rate_reduction:.0%} 
reduction in churn ({lower_churn_rate:.2%})], 
and a :red[{churn_rate_reduction:.0%} increase in churn ({upper_churn_rate:.2%})].

The takeaway? Your current ${param_starting_value:,} will become ${future_state_base:,} 
over {num_months} months, if nothing changes.
''')

st.plotly_chart(fig_proj, theme=None, config={'displayModeBar': False})
st.markdown(f"""
We think a better way to look at this is the cumulative revenue over time - the total money you stand to gain or lose
by reducing or increasing churn over the full {num_months} period, compared against your current state.

Below, you'll see the difference in the total revenue by :red[increasing] or :violet[decreasing] churn 
by {churn_rate_reduction:.0%}. **It grows exponentially over time!** This also highlights that changes *today* are 
most important for long-term success.
""")
st.plotly_chart(fig_rev, theme=None, config={'displayModeBar': False})

st.subheader(f'Is reducing churn my only tool?')
st.markdown(f'''
Great question! Thankfully, we have 3 levers to pull here:
1. Reducing churn
2. Increasing acquisitions
3. Increasing existing subscription's value
 
In our experience, you'll have the most wide-reaching success with targeting #1 and #2; #3 is challenging 
(it's an easy way to annoy your clientele - "please pay us more!") but can be executed right in some circumstances. 

Obviously, it's not a one-or-the-other situation either. You can simulatenously increase acquisitions while reducing 
churn, and both will lead to stronger outcomes. 

## {param_client_name_possessive} route to growth
Below, you'll see a heatmap showing your revenue per month
 in {num_months} months at various combinations of 
acquisition rates and churn rates. 

Anything above the dotted-red line means you're growing, anything below means you're declining. 

The takeaway? Relatively changes in churn rates have **big** impacts over the long-term.
''')

st.plotly_chart(heatmap, theme=None, config={'displayModeBar': False})

st.markdown("""
# How?

Let's talk.
""")

st.markdown("""



""", unsafe_allow_html=True)

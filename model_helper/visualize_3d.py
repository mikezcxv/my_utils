import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)


def plotmy3d(c, name):
    data = [go.Surface(z=c)]
    layout = go.Layout(title=name, autosize=False,
                       width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# plotmy3d(df[['ranking', 'comparison_rate', target]].ix[:500].values, 'test')
# plotmy3d(df[['lang_en-AU', 'lang_en-US', target]].ix[:500].values, 'test')

# plotmy3d(df[['ranking', 'domain_sessionidx', target]].ix[:1000].values, 'test')
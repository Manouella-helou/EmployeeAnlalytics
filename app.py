
# Imports
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import os
import tempfile

# Color palette
# Color palette
COLOR_PALETTE = {
    'primary': '#2C3E50',    # Dark Blue
    'success': '#27AE60',    # Green
    'danger': '#E74C3C',     # Red
    'info': '#3498DB',       # Light Blue
    'background': '#F5F6FA', # Light Gray
    'text': '#2C3E50'        # Dark Blue
}

# Data Processing Functions
def process_data(contents):
    """Process uploaded file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Read CSV file
        raw_data = io.StringIO(decoded.decode('utf-8'))
        df = pd.read_csv(raw_data)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'name': 'Employee Name',
            'statusReport': 'Status',
            'jobTitle': 'Job Title',
            'startDateReport': 'Start Date',
            'Final Manager': 'Manager'
        })
        
        # Drop empty rows
        df = df.dropna(how='all')
        
        # Ensure date columns are datetime
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        df['Termination Date'] = pd.to_datetime(df['Termination Date'], errors='coerce')
        
        # Add Month columns for analysis
        df['Start Month'] = df['Start Date'].dt.strftime('%Y-%m')
        df['Termination Month'] = df['Termination Date'].dt.strftime('%Y-%m').fillna('')
        
        # Calculate initial headcount
        october_start = pd.Timestamp('2024-10-01')
        initial_headcount = len(df[
            (df['Start Date'] < october_start) & 
            ((df['Termination Date'].isna()) | (df['Termination Date'] >= october_start))
        ])
        
        return df, initial_headcount
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_monthly_metrics(df, initial_headcount):
    """Calculate monthly hiring and termination metrics"""
    months = pd.date_range(start='2024-10-01', end='2024-11-30', freq='ME')
    monthly_data = []

    current_headcount = initial_headcount

    for month in months:
        month_start = month.replace(day=1)
        month_end = month
        
        # New hires in this month
        month_hires = len(df[
            (df['Start Date'] >= month_start) & 
            (df['Start Date'] <= month_end)
        ])

        # Terminations in this month
        month_terms = len(df[
            (df['Termination Date'] >= month_start) & 
            (df['Termination Date'] <= month_end)
        ])

        # Calculate end of month headcount
        month_end_headcount = current_headcount + month_hires - month_terms

        monthly_data.append({
            'Month': month.strftime('%Y-%m'),
            'Starting Headcount': current_headcount,
            'Hires': month_hires,
            'Terminations': month_terms,
            'Ending Headcount': month_end_headcount
        })

        current_headcount = month_end_headcount

    return pd.DataFrame(monthly_data)

def create_headcount_trend(df):
    """Create headcount trend analysis."""
    start_date = df['Start Date'].min()
    end_date = pd.Timestamp('2024-11-30')

    # Create monthly date range
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    trend_data = []

    for date in dates:
        # Count active employees at this date
        active_count = len(df[
            (df['Start Date'].notna()) & 
            (df['Start Date'] <= date) & 
            ((df['Termination Date'].isna()) | (df['Termination Date'] > date))
        ])

        trend_data.append({
            'Date': date,
            'Headcount': active_count
        })

    trend_df = pd.DataFrame(trend_data)
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['Headcount'],
        mode='lines+markers',
        name='Headcount',
        line=dict(color=COLOR_PALETTE['primary'], width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Employee Headcount Trend',
        xaxis_title='Date',
        yaxis_title='Number of Employees',
        height=400,
        showlegend=False,
        hovermode='x'
    )

    return fig

def create_summary_cards(df, initial_headcount):
    """Create summary metric cards"""
    oct_start = pd.Timestamp('2024-10-01')
    nov_end = pd.Timestamp('2024-11-30')
    
    # Calculate metrics
    total_hires = len(df[df['Start Date'].between(oct_start, nov_end)])
    total_terms = len(df[df['Termination Date'].between(oct_start, nov_end)])
    current_headcount = len(df[
        (df['Start Date'] <= nov_end) & 
        ((df['Termination Date'].isna()) | (df['Termination Date'] > nov_end))
    ])
    
    cards = [
        html.Div([
            html.H4("Initial Headcount (Oct 1st)"),
            html.H2(str(initial_headcount))
        ], className='card'),
        html.Div([
            html.H4("New Hires (Oct-Nov)"),
            html.H2(str(total_hires))
        ], className='card'),
        html.Div([
            html.H4("Terminations (Oct-Nov)"),
            html.H2(str(total_terms))
        ], className='card'),
        html.Div([
            html.H4("Current Headcount"),
            html.H2(str(current_headcount))
        ], className='card')
    ]
    
    return cards
def calculate_manager_breakdown(df):
    """Calculate the breakdown of active employees per manager"""
    # Filter for active employees
    active_employees = df[df['Status'] == 'Active']
    
    # Group by manager and collect employee names
    manager_groups = active_employees.groupby('Manager')['Employee Name'].agg(list).reset_index()
    manager_groups.columns = ['Manager', 'Employee Names']
    
    # Add count column
    manager_groups['Employee Count'] = manager_groups['Employee Names'].str.len()
    
    # Convert employee names list to comma-separated string
    manager_groups['Employee Names'] = manager_groups['Employee Names'].apply(lambda x: ', '.join(x))
    
    # Sort by count descending
    manager_groups = manager_groups.sort_values('Employee Count', ascending=False)
    
    return manager_groups



# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# App Layout
app.layout = html.Div([
    # Data Store
    dcc.Store(id='stored-data'),
    dcc.Store(id='initial-headcount'),
    
    # Header
    html.Div([
        html.H1("📊 Business Analyst Dashboard", className='dashboard-title'),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    '📁 Drag and Drop or ',
                    html.A('Select File')
                ]),
                className='upload-box'
            )
        ])
    ], className='dashboard-header'),
    
    # Main Content
    html.Div(id='dashboard-content', children=[
        # Summary Cards
        html.Div(id='summary-cards', className='summary-cards'),
        
        # Monthly Summary Table
        html.Div([
            html.H3("📅 Monthly Summary", className='section-title'),
            html.Div(id='monthly-summary-table')
        ], className='table-section'),
        
        # Headcount Trend Chart
        html.Div([
            html.H3("📈 Headcount Trend", className='section-title'),
            dcc.Graph(id='headcount-trend-chart')
        ], className='chart-section'),
        
        # Manager Distribution Charts
        html.Div([
            # Active BAs per Manager
            html.Div([
                html.H3("👥 Active BAs by Manager", className='section-title'),
                dcc.Graph(id='active-manager-chart')
            ], className='chart-section'),
            
            # Terminated BAs per Manager
            html.Div([
                html.H3("👥 Terminated BAs by Manager", className='section-title'),
                dcc.Graph(id='terminated-manager-chart')
            ], className='chart-section'),
        ], className='dashboard-grid'),
        
        # BA Details Table
        html.Div([
            html.H3("📋 BA Details", className='section-title'),
            html.Div(id='ba-table')
        ], className='table-section'),
        
        # Download Section
        html.Div([
            html.Button("📥 Download Report", id='btn-report', className='btn-primary'),
            html.Button("📊 Export to Excel", id='btn-excel', className='btn-secondary'),
            dcc.Download(id="download-report"),
            dcc.Download(id="download-excel")
        ], className='button-group')
    ], style={'display': 'none'})
], className='container')

# Add CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>BA Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background-color: ''' + COLOR_PALETTE['background'] + ''';
                margin: 0;
                padding: 20px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }
            
            .dashboard-title {
                font-size: 2.5rem;
                font-weight: 700;
                color: ''' + COLOR_PALETTE['text'] + ''';
                margin: 0;
            }
            
            .upload-box {
                border: 2px dashed ''' + COLOR_PALETTE['primary'] + ''';
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .upload-box:hover {
                background-color: rgba(44, 62, 80, 0.1);
            }
            
            .summary-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .card h4 {
                margin: 0;
                color: ''' + COLOR_PALETTE['text'] + ''';
                font-weight: 600;
            }
            
            .card h2 {
                margin: 10px 0 0 0;
                color: ''' + COLOR_PALETTE['primary'] + ''';
                font-size: 2rem;
            }
            
            .chart-section, .table-section {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .section-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: ''' + COLOR_PALETTE['text'] + ''';
                margin-bottom: 20px;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                justify-content: center;
                margin-top: 20px;
                padding: 20px;
            }
            
            .btn-primary, .btn-secondary {
                padding: 12px 24px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .btn-primary {
                background-color: ''' + COLOR_PALETTE['primary'] + ''';
                color: white;
            }
            
            .btn-primary:hover {
                background-color: #1a252f;
            }
            
            .btn-secondary {
                background-color: white;
                color: ''' + COLOR_PALETTE['primary'] + ''';
                border: 1px solid ''' + COLOR_PALETTE['primary'] + ''';
            }
            
            .btn-secondary:hover {
                background-color: ''' + COLOR_PALETTE['background'] + ''';
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div style="text-align: center; padding: 20px; color: #666; font-size: 0.75rem; margin-top: 40px; border-top: 1px solid #eee;">
            © 2024 BA Dashboard. All Rights Reserved. | Developed by Manouella Helou
    </body>
</html>
'''

# Callbacks
@app.callback(
    [Output('stored-data', 'data'),
     Output('initial-headcount', 'data'),
     Output('dashboard-content', 'style'),
     Output('monthly-summary-table', 'children'),
     Output('summary-cards', 'children'),
     Output('headcount-trend-chart', 'figure'),
     Output('active-manager-chart', 'figure'),
     Output('terminated-manager-chart', 'figure'),
     Output('ba-table', 'children')],
    [Input('upload-data', 'contents')]
)
def update_dashboard(contents):
    if contents is None:
        return None, None, {'display': 'none'}, None, None, {}, {}, {}, None
    
    try:
        # Process data
        df, initial_headcount = process_data(contents)
        if df is None:
            raise Exception("Error processing data")
        
        # Calculate monthly metrics
        monthly_df = calculate_monthly_metrics(df, initial_headcount)
        
        # Create monthly summary table
        monthly_table = DataTable(
            data=monthly_df.to_dict('records'),
            columns=[
                {"name": "Month", "id": "Month"},
                {"name": "Starting Headcount", "id": "Starting Headcount"},
                {"name": "Hires", "id": "Hires"},
                {"name": "Terminations", "id": "Terminations"},
                {"name": "Ending Headcount", "id": "Ending Headcount"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'backgroundColor': 'white'
            },
            style_header={
                'backgroundColor': COLOR_PALETTE['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
        
        # Create summary cards
        summary_cards = create_summary_cards(df, initial_headcount)
        
        # Create headcount trend chart
        trend_fig = create_headcount_trend(df)
        
        # Create active BAs per manager chart
        active_managers = df[df['Status'] == 'Active']['Manager'].value_counts()
        fig_active = px.bar(
            active_managers,
            title='Active BAs per Manager',
            labels={'value': 'Number of BAs', 'index': 'Manager'},
            color_discrete_sequence=[COLOR_PALETTE['success']]
        )
        fig_active.update_layout(
            xaxis_tickangle=45,
            height=400,
            margin=dict(b=100),
            showlegend=False
        )
        
        # Create terminated BAs per manager chart
        terminated_managers = df[df['Status'] == 'Terminated']['Manager'].value_counts()
        fig_terminated = px.bar(
            terminated_managers,
            title='Terminated BAs per Manager',
            labels={'value': 'Number of BAs', 'index': 'Manager'},
            color_discrete_sequence=[COLOR_PALETTE['danger']]
        )
        fig_terminated.update_layout(
            xaxis_tickangle=45,
            height=400,
            margin=dict(b=100),
            showlegend=False
        )
        
        # Create BA details table
        ba_table = DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Employee Name', 'id': 'Employee Name'},
                {'name': 'Manager', 'id': 'Manager'},
                {'name': 'Start Date', 'id': 'Start Date'},
                {'name': 'Status', 'id': 'Status'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': COLOR_PALETTE['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} eq "Active"'},
                    'color': COLOR_PALETTE['success'],
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} eq "Terminated"'},
                    'color': COLOR_PALETTE['danger'],
                    'fontWeight': 'bold'
                }
            ],
            page_size=10,
            sort_action='native',
            filter_action='native'
        )
        
        # Store processed data
        stored_data = df.to_dict('records')
        
        return (stored_data, initial_headcount, {'display': 'block'}, monthly_table, 
                summary_cards, trend_fig, fig_active, fig_terminated, ba_table)
        
    except Exception as e:
        print(f"Error updating dashboard: {str(e)}")
        return None, None, {'display': 'none'}, None, None, {}, {}, {}, None



@app.callback(
    Output("download-report", "data"),
    Input("btn-report", "n_clicks"),
    [State('stored-data', 'data'),
     State('initial-headcount', 'data')],
    prevent_initial_call=True
)
def generate_report(n_clicks, stored_data, initial_headcount):
    if not n_clicks or not stored_data:
        return None
    
    try:
        # Convert stored data back to DataFrame and ensure proper datetime conversion
        df = pd.DataFrame(stored_data)
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['Termination Date'] = pd.to_datetime(df['Termination Date'])
        
        # Calculate metrics and create visualizations
        monthly_df = calculate_monthly_metrics(df, initial_headcount)
        trend_data = create_headcount_trend(df)

        # Generate trend data points
        trend_data_points = []
        for trace in trend_data['data']:
            if 'x' in trace and 'y' in trace:
                for x, y in zip(trace['x'], trace['y']):
                    trend_data_points.append({'Date': pd.to_datetime(x), 'Headcount': y})
        trend_df = pd.DataFrame(trend_data_points)
        
        # Calculate executive summary data
        total_hires = monthly_df['Hires'].sum()
        total_terminations = monthly_df['Terminations'].sum()
        current_headcount = monthly_df.iloc[-1]['Ending Headcount']
        current_month_net_change = current_headcount - monthly_df.iloc[-2]['Ending Headcount'] if len(monthly_df) > 1 else current_headcount - initial_headcount

        # Create images in temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as trend_img_temp, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as active_img_temp, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as terminated_img_temp:
            
            trend_img_path = trend_img_temp.name
            active_img_path = active_img_temp.name
            terminated_img_path = terminated_img_temp.name

        # Plot trend data
        plt.figure(figsize=(10, 5))
        plt.plot(trend_df['Date'], trend_df['Headcount'], 
                 color=COLOR_PALETTE['primary'], marker='o')
        plt.title('Headcount Trend', pad=20)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(trend_img_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Plot active BAs per manager
        active_managers = df[df['Status'] == 'Active']['Manager'].value_counts()
        plt.figure(figsize=(10, 5))
        plt.bar(active_managers.index, active_managers.values, color=COLOR_PALETTE['success'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Active BAs per Manager', pad=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(active_img_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # Plot terminated BAs per manager
        terminated_managers = df[df['Status'] == 'Terminated']['Manager'].value_counts()
        plt.figure(figsize=(10, 5))
        plt.bar(terminated_managers.index, terminated_managers.values, color=COLOR_PALETTE['danger'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Terminated BAs per Manager', pad=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(terminated_img_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

        # Build PDF report with same content structure
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=50, leftMargin=50, 
                                topMargin=50, bottomMargin=50)

        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor(COLOR_PALETTE['primary'])
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor(COLOR_PALETTE['primary'])
        )

        # Create report content
        elements = [
            Paragraph("Business Analyst Report", title_style),
            Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
            Spacer(1, 20),
            Paragraph("Executive Summary", heading_style),
        ]

        # Add executive summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Initial Headcount', f"{initial_headcount:,}"],
            ['Current Headcount', f"{int(current_headcount):,}"],
            ['Total Hires (Oct-Nov)', f"{int(total_hires):,}"],
            ['Total Terminations (Oct-Nov)', f"{int(total_terminations):,}"],
            ['Current Month Net Change', f"{int(current_month_net_change):+,}"]
        ]
        summary_table = Table(summary_data, colWidths=[150, 150])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLOR_PALETTE['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        # Add headcount trend chart
        elements.append(KeepTogether([
            Paragraph("Headcount Trend", heading_style),
            Spacer(1, 10),
            Image(trend_img_path, width=450, height=250),
            Spacer(1, 20)
        ]))

        # Add monthly summary table
        elements.append(Paragraph("Monthly Summary", heading_style))
        monthly_data = [['Month', 'Starting', 'Hires', 'Terminations', 'Ending']]
        for _, row in monthly_df.iterrows():
            monthly_data.append([
                row['Month'],
                str(int(row['Starting Headcount'])),
                str(int(row['Hires'])),
                str(int(row['Terminations'])),
                str(int(row['Ending Headcount']))
            ])
        monthly_table = Table(monthly_data, colWidths=[80, 80, 80, 80, 80])
        monthly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLOR_PALETTE['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(monthly_table)
        elements.append(Spacer(1, 20))

        # Add charts
        elements.append(Paragraph("Headcount Trend", heading_style))
        elements.append(Image(trend_img_path, width=450, height=250))
        elements.append(Spacer(1, 20))
        
        elements.append(Paragraph("Manager Distribution", heading_style))
        elements.append(Image(active_img_path, width=450, height=250))
        elements.append(Spacer(1, 10))
        elements.append(Image(terminated_img_path, width=450, height=250))
        
        # Build PDF
        doc.build(elements)
        output.seek(0)

        # Clean up temporary files
        os.unlink(trend_img_path)
        os.unlink(active_img_path)
        os.unlink(terminated_img_path)

        return dcc.send_bytes(output.read(), f"ba_report_{datetime.now().strftime('%Y%m%d')}.pdf")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
@app.callback(
    Output("download-excel", "data"),
    Input("btn-excel", "n_clicks"),
    [State('stored-data', 'data'),
     State('initial-headcount', 'data')],
    prevent_initial_call=True
)
def export_excel(n_clicks, stored_data, initial_headcount):
    if not n_clicks or not stored_data:
        return None
    
    try:
        # Convert stored data back to DataFrame and ensure proper datetime conversion
        df = pd.DataFrame(stored_data)
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['Termination Date'] = pd.to_datetime(df['Termination Date'])
        
        # Calculate monthly metrics
        monthly_df = calculate_monthly_metrics(df, initial_headcount)
        
        # Calculate manager breakdown
        manager_breakdown = calculate_manager_breakdown(df)
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Get workbook and create format
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'font_color': 'white',
                'bg_color': COLOR_PALETTE['primary'],
                'border': 1
            })
            
            # Create date format
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Write monthly summary
            monthly_df.to_excel(writer, sheet_name='Monthly Summary', index=False)
            worksheet = writer.sheets['Monthly Summary']
            for col_num, value in enumerate(monthly_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_column('A:E', 18)
            
            # Write BA details
            df_excel = df.copy()
            # Convert datetime columns to string format for Excel
            df_excel['Start Date'] = df_excel['Start Date'].dt.strftime('%Y-%m-%d')
            df_excel['Termination Date'] = df_excel['Termination Date'].dt.strftime('%Y-%m-%d')
            
            df_excel.to_excel(writer, sheet_name='BA Details', index=False)
            worksheet = writer.sheets['BA Details']
            
            # Format headers
            for col_num, value in enumerate(df_excel.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 35)  # Employee Name
            worksheet.set_column('B:B', 30)  # Manager
            worksheet.set_column('C:C', 15)  # Start Date
            worksheet.set_column('D:D', 12)  # Status
            
            # Write Manager Breakdown summary
            manager_breakdown.to_excel(writer, sheet_name='Manager Breakdown', index=False)
            worksheet = writer.sheets['Manager Breakdown']
            for col_num, value in enumerate(manager_breakdown.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_column('A:A', 20)  # Manager
            worksheet.set_column('B:B', 15)  # Employee Count
            worksheet.set_column('C:C', 50)  # Employee Names
            
            # Create individual sheets for each manager
            for manager in df['Manager'].unique():
                # Filter data for this manager
                manager_data = df[df['Manager'] == manager].copy()
                
                # Convert datetime columns for this manager's data
                manager_data['Start Date'] = manager_data['Start Date'].dt.strftime('%Y-%m-%d')
                manager_data['Termination Date'] = manager_data['Termination Date'].dt.strftime('%Y-%m-%d')
                
                # Create sheet name (limit length to avoid Excel errors)
                sheet_name = f"{manager[:30]}" if len(manager) > 30 else manager
                
                # Add manager statistics at the top
                active_count = len(df[
                    (df['Manager'] == manager) & 
                    (df['Status'] == 'Active')
                ])
                terminated_count = len(df[
                    (df['Manager'] == manager) & 
                    (df['Status'] == 'Terminated')
                ])
                
                # Create statistics DataFrame
                stats_df = pd.DataFrame([
                    ['Active Employees', active_count],
                    ['Terminated Employees', terminated_count],
                    ['Total Employees', len(manager_data)]
                ], columns=['Metric', 'Count'])
                
                # Write statistics
                stats_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                
                # Write employee details below statistics
                manager_data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=5)
                
                # Format sheet
                worksheet = writer.sheets[sheet_name]
                
                # Format statistics header
                for col_num, value in enumerate(stats_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Format employee details header
                for col_num, value in enumerate(manager_data.columns.values):
                    worksheet.write(5, col_num, value, header_format)
                
                # Set column widths
                worksheet.set_column('A:A', 35)  # Employee Name
                worksheet.set_column('B:B', 30)  # Manager
                worksheet.set_column('C:C', 15)  # Start Date
                worksheet.set_column('D:D', 12)  # Status
                
                # Add conditional formatting for status
                active_format = workbook.add_format({'color': COLOR_PALETTE['success'], 'bold': True})
                terminated_format = workbook.add_format({'color': COLOR_PALETTE['danger'], 'bold': True})
                
                status_col = df_excel.columns.get_loc('Status')
                worksheet.conditional_format(6, status_col, 
                                          5 + len(manager_data), status_col,
                                          {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Active',
                                           'format': active_format})
                
                worksheet.conditional_format(6, status_col, 
                                          5 + len(manager_data), status_col,
                                          {'type': 'text',
                                           'criteria': 'containing',
                                           'value': 'Terminated',
                                           'format': terminated_format})
        
        output.seek(0)
        return dcc.send_bytes(output.read(),
                            f"ba_data_{datetime.now().strftime('%Y%m%d')}.xlsx")
        
    except Exception as e:
        print(f"Error exporting to Excel: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
if __name__ == '__main__':
    # Get port from environment variable or use 8000 as default
    port = int(os.getenv("PORT", 8000))
    app.run_server(host='0.0.0.0', port=port, debug=False)

server = app.server  # This is needed for Koyeb

'''
Date: 29 March 2022

Description: design of dashboard to simulate the spread of COVID-19 inside a supermarket.

Functionality: canvas to create a supermarket layout, input of the parameters of the model, run simulation and display visualization results

Parameter: number of customers, with or without facemask, supermarket's entrance and exits

Output: visualization results
'''

from re import A
import dash
import dash_daq as daq
#from dash import html,dcc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output     # link the dropdown values with the graph

from ipycanvas import Canvas
from dash_canvas import DashCanvas

from dash_canvas.utils import (parse_jsonstring,
                              superpixel_color_segmentation,
                              image_with_contour, image_string_to_PILImage,
                              array_to_data_url)

from skimage import io

import os

import shutil
from pathlib import Path

import PIL
from PIL import Image

import shutil

#import cv2
import glob

import math

import base64

from Simulation import *

from selenium import webdriver

import os.path
from os import path

import math
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#plt.style.use('seaborn-deep')
import numpy as np

import sys


# import the external CSS style for the layout of the dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) 
#app = dash.Dash()

# variables for of creation of supermarket layout in the canvas 
supermarket01_draw_config = {
    "modeBarButtonsToAdd": [
        "drawrect",
        "eraseshape",
    ],
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'supermarket01',
        'height': 909,
        'width': 909,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'scrollZoom': True
}

supermarket02_draw_config = {
    "modeBarButtonsToAdd": [
        "drawrect",
        "eraseshape",
    ],
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'supermarket02',
        'height': 909,
        'width': 909,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'scrollZoom': True
}

img_rgb = np.array([[[255, 255, 255]]*909]*909, dtype=np.uint8)
supermarket01_fig = px.imshow(img_rgb)
supermarket01_fig.update_layout(autosize=False, width=909, height=909)
supermarket01_fig.update_layout(dragmode="drawrect", newshape=dict(fillcolor="black", opacity=1, line=dict(color="black", width=0)))
supermarket01_fig.update_layout(xaxis_visible= False, yaxis_visible=False, coloraxis_showscale=False)
supermarket01_fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
#supermarket01_fig.update_layout({'margin': {'l': 1, 'r': 1, 'b': 1, 't': 1, 'pad': 1},})
                
img_rgb = np.array([[[255, 255, 255]]*909]*909, dtype=np.uint8)
supermarket02_fig = px.imshow(img_rgb)
supermarket02_fig.update_layout(autosize=False, width=909, height=909)
supermarket02_fig.update_layout(dragmode="drawrect", newshape=dict(fillcolor="black", opacity=1, line=dict(color="black", width=0)))
supermarket02_fig.update_layout(xaxis_visible= False, yaxis_visible=False, coloraxis_showscale=False)
supermarket02_fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
#supermarket02_fig.update_layout({'margin': {'l': 1, 'r': 1, 'b': 1, 't': 1, 'pad': 1},})


# CSS styles for the layout of the dashboard
tabs_styles = {
    'height': '44px',
    'border-top-left-radius': '10px',
    'border-top-right-radius': '10px',
    #'backgroundColor':'#fafbfc'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-top-left-radius': '10px',
    'border-top-right-radius': '10px',
    'backgroundColor':'#fafbfc',
    'color':'#586069'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-top-left-radius': '10px',
    'border-top-right-radius': '10px',
}

#
#   Block of Python Dash 
#
app.layout = html.Div([

    #html.Meta(httpEquiv="refresh",content="1"),
    dcc.Location(id='url', refresh=False),

    #
    # Title of the dashboard
    #
    html.Div(''),
    html.H1('COVID-19 Social Distancing Simulation: Supermarket Case Study'),

    #
    #   Creation of tabs fo the dashboard
    #
    # "presistence=True" is the argument to 'anchor' or 'fix' the selected tab when reloading the dashboard
    dcc.Tabs(id='tabs', persistence=True, children=[
        
        #
        # tab 0: user manual (information for the user of this dashboard)
        #
        dcc.Tab(label='Explanation', id='tab00', style=tab_style, selected_style=tab_selected_style, children=[
            html.H2(['User manual'], style={'padding':'0%'}),
            html.P('• This is the visualiozation tool to simulate the COVID-19 spread mechanisms in a supermarket.'),
            html.P('• The user create the layout of the supermarket.'),
            html.P('• IMPORTANT: The name of images file must be supermarket01 and supermarket02 (e.g. supermarket01.png, supermarket01.jpeg, supermarket01.pbm for left supermarket layout, supermarket02.png, supermarket02.jpeg, supermarket02.pbm for right supermarket layout.).'),
            html.P('• Enter the number of customers expected to be visited.'),
            html.P('• Enter the values of the parameters for the model.'),
            html.P('• Press START SIMULATION button for running the simulation.'),
            html.P('• Press VISUALIZE button for displaying the visualizations.'),
        ]),

        #
        # tab 1.1: creation of supermarket layouts (two layouts can be created) on the canvas
        #
        dcc.Tab(label='Step 1', id='tab01', style=tab_style, selected_style=tab_selected_style, children=[
            html.H3('1.1: Create supermarket layouts'),
            html.P('These images will be saved in the download folder.'),
            html.P('Rename these images if neccessary and then copy them to the project folder.'),
            html.P('This is an OPTIONAL step if the supermarket layout images do not already exist. Please rename these images and then copy them to the project folder.'),
            html.Div([
                html.Div([
                    html.Div(
                        id="create_supermarket01",
                        children=dcc.Graph(figure=supermarket01_fig, config=supermarket01_draw_config),
                    ),
                ], className = 'five columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10}),
                
                html.Div([
                    html.Div(
                        id="create_supermarket02",
                        children=dcc.Graph(figure=supermarket02_fig, config=supermarket02_draw_config),
                    ),  
                ], className = 'five columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10}),
            ], className="row"),


            #
            # tab 1.2: convert the image (jpg, jpeg, or png) to pbm format
            #
            html.H3(['1.2: Convert and resize images to pbm format'], className = 'row'),       # style={'padding-top':'1%'
            html.P('Note: the name of image file must be supermarket01.png, supermarket01.jpg, supermarket01.jpeg for left supermarket, supermarket02.png, supermarket02.jpg, supermarket02.jpeg for right supermarket.'),
            html.P('The images are resized to width 101 pixels and height 101 pixels.'),
            html.P('Press CONVERT IMAGE to convert the images to pbm format.'),
            html.P('IMPORTANT: Please check the project folder to make sure the images have been converted successfully.'),

            html.Div([
                #dcc.ConfirmDialogProvider(id="image_converted", message="Images have been converted and resized."),
                html.Div([
                    html.P('for supermarket layout 01:'),
                    dcc.RadioItems(['png', 'jpg', 'jpeg', 'pbm'], 'png', id='image01', inline=True)
                ], className = 'three columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan',}),
                #html.Button('Convert image', id='convertImage01', n_clicks=0),
                html.Div([
                    html.P('for supermarket layout 02:'),
                    dcc.RadioItems(['png', 'jpg', 'jpeg', 'pbm'], 'png', id='image02', inline=True)
                ], className = 'three columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
                #html.Button('Convert image', id='convertImage01', n_clicks=0),
            ], className="row", style = {'padding' : '1%'}),

            dcc.ConfirmDialogProvider(
                children=html.Button('Convert image'), id='convertImage', message="Please check the folder whether images have been converted and resized.", submit_n_clicks=0),      # n_clicks=0

        ]),

        #
        # tab 2: enter the parameters to be used in the model for running the simulation
        #
        dcc.Tab(label='Step 2', id='tab03', style=tab_style, selected_style=tab_selected_style, children=[
            html.H3('2.1: Enter the number of visited customers for the supermarket layout 01 and 02'),
            
            #
            # enter the number of visited customers in the supermarkets
            # the user can enter two different number of visited customers for both different supermarket layouts
            #
            html.Div([
                html.Div([
                    html.P('Number of visited customers for supermarket layout 01 (>= 400):'),
                    dcc.Input(
                        id="num_Customer_01", type="number", value=2000, placeholder="2000",
                        min=0, max=10000, step=1,
                    ),
                ], className = 'four columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
        
                html.Div([
                    html.P('Number of visited customers for supermarket layout 02 (>= 400):'),
                    dcc.Input(
                        id="num_Customer_02", type="number", value=2000, placeholder="2000",
                        min=0, max=10000, step=1,
                    ),
                ], className = 'four columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
            ], className="row", style = {'padding' : '1%'}),

            #
            # enter the coordinates of the entrance and exits for the supermarket
            # the entrance and the exits are the same for both supermarket such that comparison is meaningful
            #
            html.H3(['2.2: Simulation parameters (supermarket entrance and exits)'], className = 'row', style={'padding-top':'1%'}),
                        
            html.Div([
                html.Div([
                    html.P('for supermarket layout 01:'),
                    html.Div([
                        html.P('Number of exits (NEXITS):'),
                        dcc.Input(id="n_exits_01", type="number", value=5, placeholder="5",),   
                    ], className = "two columns"),
                    html.Div([
                        html.P('Distance between exits in pixel (CASHIERD):'),
                        dcc.Input(id="distance_exits_01", type="number", value=14, placeholder="14",),
                    ], className = "three columns"),
                    html.Div([
                        html.P('x coordinate of the first cashier in pixel (EXITPOS):'),
                        dcc.Input(id="x_coord_exit_01", type="number", value=1, placeholder="1",),
                    ], className = "three columns"),
                    html.Div([
                        html.P('x coordinate of entrance in pixel (ENTRANCEPOS):'),
                        dcc.Input(id="x_coord_entrace_01", type="number", value=0, placeholder="0",),
                    ], className = "three columns"),
                ], className = 'row'),

                html.Div([
                    html.P('for supermarket layout 02:'),
                    html.Div([
                        html.P('Number of exits (NEXITS):'),
                        dcc.Input(id="n_exits_02", type="number", value=5, placeholder="5",),   
                    ], className = "two columns"),
                    html.Div([
                        html.P('Distance between exits in pixel (CASHIERD):'),
                        dcc.Input(id="distance_exits_02", type="number", value=14, placeholder="14",),
                    ], className = "three columns"),
                    html.Div([
                        html.P('x coordinate of the first cashier in pixel (EXITPOS):'),
                        dcc.Input(id="x_coord_exit_02", type="number", value=1, placeholder="1",),
                    ], className = "three columns"),
                    html.Div([
                        html.P('x coordinate of entrance in pixel (ENTRANCEPOS):'),
                        dcc.Input(id="x_coord_entrace_02", type="number", value=0, placeholder="0",),
                    ], className = "three columns"),
                ], className = 'row'),

                #html.Div([
                #    html.P('Number of exits (NEXITS):'),
                #    dcc.Input(id="n_exits", type="number", value=5, placeholder="5",),
                #], className = "three columns", style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
                #html.Div([
                #    html.P('Distance between exits (CASHIERD):'),
                #    dcc.Input(id="distance_exits", type="number", value=14, placeholder="14",),
                #], className = "three columns", style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
                #html.Div([
                #    html.P('x coordinate of the first cashier (EXITPOS):'),
                #    dcc.Input(id="x_coord_exit", type="number", value=1, placeholder="1",),
                #], className = "three columns", style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
                #html.Div([
                #    html.P('x coordinate of entrance (ENTRANCEPOS):'),
                #    dcc.Input(id="x_coord_entrace", type="number", value=0, placeholder="0",),
                #], className = "three columns", style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
            ], className = 'row', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),


            #
            # Make selection of without or with facemask
            # The variables to mimic the without or with facemask are set in the callback function
            # and then then these variables are used in the model
            #
            html.H3(['2.3: Simulation parameters (to mimic without or with facemask wearing scenario).'], className = 'row', style={'padding-top':'1%'}),
            
            html.Div([
                html.Div([
                    html.P('for supermarket layout 01:'),
                    dcc.RadioItems(['no mask', 'surgical mask', 'N95 mask'], 'no mask', id='mask_01', inline=True)
                ], className = 'three columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan',}),

                html.Div([
                    html.P('for supermarket layout 02:'),
                    dcc.RadioItems(['no mask', 'surgical mask', 'N95 mask'], 'no mask', id='mask_02', inline=True)
                ], className = 'three columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan',}),
            ], className="row", style = {'padding' : '1%'}),

            #
            # enter the parameters value to mimic the ventilation in the supermarket
            # these parameters are (1) diffusion coefficient, (2) sink coefficient and (3) plume coefficient
            #
            #html.Div([
            #    html.Div([
            #        html.Div([
            #            html.P('Diffusion coefficient for supermarket layout 01 (DIFFCOEFF):'),
            #            html.P(['Note: Lower the value (e.g. 0.0005) to mimic the facemask wearing situation, whereas higher value (e.g. 0.05) to mimic no wearing of facemask inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="aerosol_01", type="number", value=0.05, placeholder="0.05",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),
            #
            #        html.Div([
            #            html.P('Sink coefficient for supermarket layout 01 (ACSINKCOEFF):'),
            #            html.P(['Note: Lower the value (e.g. 0.01) to mimic the no wearing facemask situation, whereas higher value (e.g. 0.10) to mimic facemask wearing inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="sink_01", type="number", value=0.01, placeholder="0.01",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),    
            #
            #        html.Div([
            #            html.P('Plume coefficient for supermarket layout 01 (PLUMEMIN):'),
            #            html.P(['Note: Lower the value (e.g. 0.0) to mimic the no wearing facemask situation, whereas higher value (e.g. 1.0) to mimic facemask wearing inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="plume_01", type="number", value=0.0, placeholder="0.0",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),    
            #    ], className = 'twelve columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
            #
            #    html.Div([
            #        html.Div([
            #            html.P('Diffusion coefficient for supermarket layout 02 (DIFFCOEFF):'),
            #            html.P(['Note: Lower the value (e.g. 0.0005) to mimic the facemask wearing situation, whereas higher value (e.g. 0.05) to mimic no wearing of facemask inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="aerosol_02", type="number", value=0.05, placeholder="0.05",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),
            #
            #        html.Div([
            #            html.P('Sink coefficient for supermarket layout 02 (ACSINKCOEFF):'),
            #            html.P(['Note: Lower the value (e.g. 0.01) to mimic the no wearing facemask situation, whereas higher value (e.g. 0.10) to mimic facemask wearing inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="sink_02", type="number", value=0.01, placeholder="0.01",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),    
            #
            #        html.Div([
            #            html.P('Plume coefficient for supermarket layout 02 (PLUMEMIN):'),
            #            html.P(['Note: Lower the value (e.g. 0.0) to mimic the no wearing facemask situation, whereas higher value (e.g. 1.0) to mimic facemask wearing inside the supermarket.'], className = 'row', style={'padding-top':'1%'}),
            #            dcc.Input(
            #                id="plume_02", type="number", value=0.0, placeholder="0.0",
            #                min=0, max=100, step=0.0000001,
            #            ),
            #        ], className = 'four columns'),    
            #    ], className = 'twelve columns', style = {'padding' : '1%', 'border':'2px black solid', 'border-radius': 10, 'backgroundColor':'lightcyan'}),
            #
            #], className="row", style = {'padding' : '1%'}),


            html.H6('Figure below shows the parameters.', style = {'padding-top' : '1%'}),
            html.Div([
                html.Img(src=app.get_asset_url('supermarket_layout_parameters.jpeg'), style={'height':'60%', 'width':'60%'}),
            ], className = 'six columns'),
        ]),

        #
        # tab 3: button to start the model simulation
        #
        dcc.Tab(label='Step 3', id='tab05', style=tab_style, selected_style=tab_selected_style, children=[
            html.H3(['Start simulation'], className = 'row', style={'padding-top':'1%'}),
            html.P('Press the START SIMULATION button to start the simulation.'),
            dcc.ConfirmDialogProvider(
                children=html.Button('Start Simulation'), id='simulation', message="Simulation is started. Please check the terminal for the simulation status. Message \"All customers have visited the store\" will be shown when the simulation is completed.", submit_n_clicks=0),
            
            html.H6('Figure below shows the example of outputs in the terminal', style = {'padding-top' : '1%'}),
            html.P('The printed message "All customers have visited the store" means the simulation is completed for a supermarket layout.'),
            html.Div([
                html.Img(src=app.get_asset_url('example_output_terminal.jpeg'), style={'height':'80%', 'width':'80%'}),
            ], className = 'six columns'),
        ]),

        #
        # tab 4: visualization of the heatmaps in the normalized time steps
        # the user can select the normalized time step range with the slider
        #
        dcc.Tab(label='Visualization I', id='viz01', style=tab_style, selected_style=tab_selected_style, children=[

            html.H2(['Visualization (heatmaps)'], className = 'row', style={'padding-top':'1%', 'font-weight': 'bold'}),
            html.P('For comparison of the heatmaps between supermarket layout 01 and 02.'),
            html.P('Range sliders are provided for displaying the normalized range step of heatmaps.'),
            html.P('0.00 means beginning of the supermarket business day.'),
            html.P('0.50 means half-day of the supermarket business day.'),
            html.P('1.00 means ending of the supermarket business day.'),
            html.P('IMPORTANT: refresh the webpage to re-load the visualization results.'),

            html.Div(style={'padding': '1%'}, children=[
                html.Div([
                    dcc.ConfirmDialogProvider(
                        children=html.Button('Visualize supermarket 01'), id='viz_supermarket_01', message='Generation of visualizations are started. Please check the assets folder and refresh.', submit_n_clicks=0),
                ], className = 'six columns'),
                html.Div([
                    dcc.ConfirmDialogProvider(
                        children=html.Button('Visualize supermarket 02'), id='viz_supermarket_02', message='Generation of visualizations are started. Please check the assets folder and refresh.', submit_n_clicks=0),
                ], className = 'six columns'),
            ], className="row"),

            html.Div([
                html.Div([
                    html.P('Number images per row (supermarket 01):'),
                    dcc.Input(
                        id="imageRow_01", type="number", value=10, placeholder="10",
                        min=1, max=15, step=1, persistence=True,
                    ),
                ], className = 'six columns'),
        
                html.Div([
                    html.P('Number images per row (supermarket 02):'),
                    dcc.Input(
                        id="imageRow_02", type="number", value=10, placeholder="10",
                        min=1, max=15, step=1, persistence=True,
                    ),
                ], className = 'six columns'),
            ], className="row", style = {'padding' : '1%'}),


            html.Div(style={'padding': '1%'}, children=[
                html.Div([
                    #dcc.RangeSlider(min=0, max=maxRange01, marks=None, value=[0, math.floor(maxRange01/6)], id='slider_supermarket_01',
                    dcc.RangeSlider(0.00, 1.00, step=0.01, marks={0.00:'0.00', 0.25:'0.25', 0.50:'0.50', 0.75:'0.75', 1.00:'1.00'}, value=[0.00, 0.25], id='slider_supermarket_01', persistence=True,
                        tooltip={"placement": "bottom", "always_visible": True}),
                ], className = 'six columns'),
                html.Div([
                    #dcc.RangeSlider(min=0, max=maxRange02, marks=None, value=[0, math.floor(maxRange02/6)], id='slider_supermarket_02',
                    dcc.RangeSlider(0.00, 1.00, step=0.01, marks={0.00:'0.00', 0.25:'0.25', 0.50:'0.50', 0.75:'0.75', 1.00:'1.00'}, value=[0.00, 0.25], id='slider_supermarket_02', persistence=True,
                        tooltip={"placement": "bottom", "always_visible": True}),
                ], className = 'six columns'),
            ], className="row"),

            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url('colorbar_horizontal.png'), style={'height':'40%', 'width':'40%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
                html.Div([
                    html.Img(src=app.get_asset_url('colorbar_horizontal.png'), style={'height':'40%', 'width':'40%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
            ], className="row", style = {'padding' : '1%'}),

            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url('viz_supermarket_01.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
                html.Div([
                    html.Img(src=app.get_asset_url('viz_supermarket_02.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
            ], className="row", style = {'padding' : '1%'}),
        ]),

        #
        # tab 5: visualization of the summary heatmaps (of all time steps) for both supermarkets
        #
        dcc.Tab(label='Visualization II', id='viz02', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2(['Visualization (heatmap)'], className = 'row', style={'padding-top':'1%', 'font-weight': 'bold'}),
            html.P('For comparison of the summary of heatmaps between supermarket layout 01 and 02.'),
            html.P('IMPORTANT: refresh the webpage to re-load the visualization results.'),

            html.Div(style={'padding': '1%'}, children=[
                html.Div([
                    dcc.ConfirmDialogProvider(
                        children=html.Button('Visualize supermarket'), id='viz_II_supermarket', message='Generation of visualizations are started. Please check the assets folder and refresh.', submit_n_clicks=0),
                ], className = 'six columns'),
            ], className="row"),

            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url('heatMap01.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
                html.Div([
                    html.Img(src=app.get_asset_url('heatMap02.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
            ], className="row", style = {'padding' : '1%'}),

        ]),

        #
        # tab 5: lineplot and histogram visualizations of of the model simulations
        # lineplot: Number of customers inside the supermarket in every time step (the time steps have been normalized (between 0.0 and 1.0))
        # Histogram: Number of customers exposed to different aerosol concentration levels
        #
        # Level 1: expose to between 1.0 and 5.0 aerosols/m^3. Low risk.
        # Level 2: expose to between 5.0 and 10.0 aerosols/m^3
        # Level 3: expose to between 10.0 and 50.0 aerosols/m^3
        # Level 4: expose to >= 50.0 aerosols/m^3. High risk
        #
        dcc.Tab(label='Visualization III', id='viz03', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.H2(['Visualization'], className = 'row', style={'padding-top':'1%', 'font-weight': 'bold'}),
            html.P('For comparison of the histograms and lineplots between supermarket layout 01 and 02.'),
            html.P('Lineplot: Number of customers inside the supermarket in every time step (the time steps have been normalized (between 0.0 and 1.0)).'),
            html.P('Histogram: Number of customers exposed to different aerosol concentration levels.'),
            html.P('Level 1: expose to between 1.0 and 5.0 aerosols/m^3. Low risk.'),
            html.P('Level 2: expose to between 5.0 and 10.0 aerosols/m^3.'),
            html.P('Level 3: expose to between 10.0 and 50.0 aerosols/m^3.'),
            html.P('Level 4: expose to >= 50.0 aerosols/m^3. High risk.'),
            html.P('IMPORTANT: refresh the webpage to re-load the visualization results.'),

            html.Div(style={'padding': '1%'}, children=[
                html.Div([
                    dcc.ConfirmDialogProvider(
                        children=html.Button('Visualize supermarket'), id='viz_III_supermarket', message='Generation of visualizations are started. Please check the assets folder and refresh.', submit_n_clicks=0),
                ], className = 'six columns'),
            ], className="row"),

            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url('numCustomers.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
                html.Div([
                    html.Img(src=app.get_asset_url('numCustomersLevels.png'), style={'height':'100%', 'width':'100%', 'border':'2px black solid', 'border-radius': 8, 'backgroundColor':'white',}),
                ], className = 'six columns'),
            ], className="row", style = {'padding' : '1%'}),

        ]),
    ]),

    #
    # dummy division (for returning NONE while callback function)
    #
    html.Div(id='dummy1'),
    html.Div(id='dummy2'),
    html.Div(id='dummy3'),
    html.Div(id='dummy4'),
    html.Div(id='dummy5'),
    html.Div(id='dummy6'),
    html.Div(id='dummy7'),
])

## ============================================================ ##
## ==================== Callback functions ==================== ##
## ============================================================ ##

#
# callback function to convert the image (jpg, jpeg, or png) to pbm format
#
@app.callback(
    Output('dummy1', "children"),
    #Output('image_01', 'src'),
    Input('image01', 'value'),
    Input('image02', 'value'),
    #Input('convertImage', 'n_clicks'),
    Input('convertImage', 'submit_n_clicks'),
)
def convertImage(image01_type, image02_type, submit_n_clicks):
    # Define paths and initial settings
    downloads_path = str(Path.home() / "Downloads")
    cwd = os.getcwd()

    width_px = 101      # pixels
    height_px = 101     # pixels

    #width_px = 900      # pixels
    #height_px = 900     # pixels

    # Define file names based on selected image types
    # for supermarket layout 01
    if (image01_type == 'png'):
        file_name_01 = 'supermarket01.png'
        if(os.path.exists(downloads_path + "/supermarket01.png")):
            shutil.copy(downloads_path + "/supermarket01.png", cwd)
    elif (image01_type == 'jpg'):
        file_name_01 = 'supermarket01.jpg'
    elif (image01_type == 'jpeg'):
        file_name_01 = 'supermarket01.jpeg'
    elif (image01_type == 'pbm'):
        file_name_01 = 'supermarket01.pbm'

     # for supermarket layout 02
    if (image02_type == 'png'):
        file_name_02 = 'supermarket02.png'
        if(os.path.exists(downloads_path + "/supermarket02.png")):
            shutil.copy(downloads_path + "/supermarket02.png", cwd)
    elif (image02_type == 'jpg'):
        file_name_02 = 'supermarket02.jpg'
    elif (image02_type == 'jpeg'):
        file_name_02 = 'supermarket02.jpeg'
    elif (image02_type == 'pbm'):
        file_name_02 = 'supermarket02.pbm'

    
    try:
        # Open the selected images
        img01 = Image.open(file_name_01)
        img02 = Image.open(file_name_02)
    except:
        # Handle the case where one of the image files does not exist
        print("One of the image files do not exist (e.g. 01.png, 0.2.jpg, etc.).") # Handle the error accordingly
        #return None

    if (submit_n_clicks > 0):
        cwd = os.getcwd()
        print("current directory: ", cwd)

        # Convert images to PBM format, resize, and save
        img_pbm_01 = img01.convert('1')
        img_pbm_01_resize = img_pbm_01.resize((width_px, height_px))
        img_pbm_01_resize.save('supermarket01.pbm')

        img_pbm_02 = img02.convert('1')
        img_pbm_02_resize = img_pbm_02.resize((width_px, height_px))
        img_pbm_02_resize.save('supermarket02.pbm')

        cwd = os.getcwd()
        print("current directory: ", cwd)

    return None

#
# callback function to run the model simulation
#
@app.callback(
    Output('dummy2', "children"),       # temporary (this one might need to be change to plot the visualization)
    Input('n_exits_01', 'value'),
    Input('distance_exits_01', 'value'),
    Input('x_coord_exit_01', 'value'),
    Input('x_coord_entrace_01', 'value'),
    Input('n_exits_02', 'value'),
    Input('distance_exits_02', 'value'),
    Input('x_coord_exit_02', 'value'),
    Input('x_coord_entrace_02', 'value'),
    Input('num_Customer_01', 'value'),
    Input('num_Customer_02', 'value'),
    #Input('aerosol_01','value'),
    #Input('aerosol_02','value'),
    #Input('sink_01','value'),
    #Input('sink_02','value'),
    #Input('plume_01','value'),
    #Input('plume_02','value'),
    Input('mask_01','value'),
    Input('mask_02','value'),
    Input('simulation', 'submit_n_clicks')
)
def simulation(n_exits_01, distance_exits_01, x_coord_exit_01, x_coord_entrace_01, n_exits_02, distance_exits_02, x_coord_exit_02, x_coord_entrace_02, num_Customer_01, num_Customer_02, mask_01_type, mask_02_type, submit_n_clicks):
#def simulation(n_exits, distance_exits, x_coord_exit, x_coord_entrace, num_Customer_01, num_Customer_02, aerosol_01, aerosol_02, sink_01, sink_02, plume_01, plume_02, submit_n_clicks):
    ## some params related to exits
    NEXITS_01 = n_exits_01 # how many cashiers in the store
    CASHIERD_01 = distance_exits_01 # distance between cashiers
    ENTRANCEPOS_01 = x_coord_entrace_01 # x-coord of entrance to the store
    EXITPOS_01 = x_coord_exit_01 # Lx - y-coord of the first cashier 

    ## some params related to exits
    NEXITS_02 = n_exits_02 # how many cashiers in the store
    CASHIERD_02 = distance_exits_02 # distance between cashiers
    ENTRANCEPOS_02 = x_coord_entrace_02 # x-coord of entrance to the store
    EXITPOS_02 = x_coord_exit_02 # Lx - y-coord of the first cashier 

    numCustomer01 = num_Customer_01 # total number of customers for the LEFT supermarket layout
    numCustomer02 = num_Customer_02 # total number of customers for the RIGHT supermarket layout

    #DIFFCOEFF01 = aerosol_01    # 0.05
    #DIFFCOEFF02 = aerosol_02    # 0.0005
    DIFFCOEFF01 = 0.05    # 0.05
    DIFFCOEFF02 = 0.05    # 0.0005

    #sink_01 = 1e-2  # 0.01
    #sink_02 = 0.50             # round 01 (testing)
    #sink_02 = 0.10              # round 02 (testing)
    #ACSINKCOEFF01 = sink_01
    #ACSINKCOEFF02 = sink_02
    ACSINKCOEFF01 = 0.01
    ACSINKCOEFF02 = 0.01

    #plume_01 = 0.0
    #plume_02 = 5.0             # round 01 (testing)
    #plume_02 = 1.0              # round 02 (testing)
    #PLUMEMIN01 = plume_01
    #PLUMEMIN02 = plume_02
    PLUMEMIN01 = 0.0
    PLUMEMIN02 = 0.0

    ## PLUMECONCINC: aerosol concentration in coughing event
    ## PLUMECONCCONT: continuous aerosol emission
    # for supermarket 01
    if (mask_01_type == 'no mask'):
        PLUMECONCINC01 = 40000.0
        PLUMECONCCONT01 = 5.0 
    elif (mask_01_type == 'surgical mask'):
        PLUMECONCINC01 = 10000.0
        PLUMECONCCONT01 = 1.1
    elif (mask_01_type == 'N95 mask'):
        PLUMECONCINC01 = 24000.0
        PLUMECONCCONT01 = 1.3
    
    # for supermarket 02
    if (mask_02_type == 'no mask'):
        PLUMECONCINC02 = 40000.0
        PLUMECONCCONT02 = 5.0 
    elif (mask_02_type == 'surgical mask'):
        PLUMECONCINC02 = 10000.0
        PLUMECONCCONT02 = 1.1
    elif (mask_02_type == 'N95 mask'):
        PLUMECONCINC02 = 24000.0
        PLUMECONCCONT02 = 1.3

    # These two have to be matched as the pixel count (in x- and y-axis) of a .ppm image, if that is imported as the simulation geometry!
    pixNx = 101
    pixNy = 101

    if (submit_n_clicks > 0):
    #if (submit_n_clicks == 1):
        # create two new folders for storing the generated images
        cwd = os.getcwd()
        print("current directory: ", cwd)

        # Create the folder 01 for storing the heatmaps of supermarket layout 01
        dir = os.path.join(cwd, "01")
        if (os.path.isdir('01')):
            shutil.rmtree(dir)
        if not os.path.exists(dir):
            os.mkdir(dir)

        # Create the folder 01 for storing the heatmaps of supermarket layout 02
        dir = os.path.join(cwd, "02")
        if (os.path.isdir('02')):    
            shutil.rmtree(dir)
        if not os.path.exists(dir):
            os.mkdir(dir)

        # Run the simulation model of supermarket layout 01
        if os.path.exists("01"):
            os.chdir("01")
            ## parameters: seed, pixNx, pixNy, N shelves (if no file provided), N customers, ..
            sim = Simulation(888892, pixNx, pixNy, 25, DIFFCOEFF01, ACSINKCOEFF01, PLUMEMIN01, PLUMECONCINC01, PLUMECONCCONT01, NEXITS_01, CASHIERD_01, ENTRANCEPOS_01, EXITPOS_01, numCustomer01, outputLevel=1, maxSteps=100000, probInfCustomer=0.01, probNewCustomer=0.2,imageName="../supermarket01.pbm",useDiffusion=1,dx=1.0)
            sim.runSimulation()
            Counter.counter_img = 0
            os.chdir("..")
        
        # Run the simulation model of supermarket layout 02
        if os.path.exists("02"):
            os.chdir("02")
            Counter.counter_img = 0
            ## parameters: seed, pixNx, pixNy, N shelves (if no file provided), N customers, ..
            sim = Simulation(888892, pixNx, pixNy, 25, DIFFCOEFF02, ACSINKCOEFF02, PLUMEMIN02, PLUMECONCINC02, PLUMECONCCONT02, NEXITS_02, CASHIERD_02, ENTRANCEPOS_02, EXITPOS_02, numCustomer02, outputLevel=1, maxSteps=100000, probInfCustomer=0.01, probNewCustomer=0.2,imageName="../supermarket02.pbm",useDiffusion=1,dx=1.0)
            sim.runSimulation()
            os.chdir("..")

    return None

#
# callback function to generate the visualization of heatmaps (within the time step range) for supermarket layout 01
#
@app.callback(
    Output('dummy4', 'children'),
    [Input('slider_supermarket_01', 'value'),
    Input('imageRow_01', 'value'),
    Input('viz_supermarket_01', 'submit_n_clicks')],
)
def vizSuperMarket01(value, rowValue, submit_n_clicks):
    print(value[0])
    print(value[1])
    
    if (submit_n_clicks > 0):
        cwd = os.getcwd()
        print("current directory: ", cwd)

        dir = os.path.join(cwd, "assets")

        #imageDir = ["01", "02"]
        if os.path.exists("01"):
            os.chdir("01")

            imdir = os.getcwd()
            print("current directory: ", imdir)

            imagesPerRow = rowValue     # number of images per row

            numImage = []
            for imageFile in glob.glob("**/*.png", recursive=True):
                numImage.append(imageFile)          # read all required images for generating the new image
            numOfImages = len(numImage)

            #if value[0] > numOfImages:
            #    value[0] = numOfImages-10
            #if value[1] > numOfImages:
            #    value[1] = numOfImages
            #if value[0] > value[1]:
            #    value[0] = numOfImages-10
            #    value[1] = numOfImages

            figureX = int(value[0] * (numOfImages-1))   # convert to the figure number
            figureY = int(value[1] * (numOfImages-1))   # convert to the figure number

            images_00 = []
            for i in range(figureX, figureY, 1):
                fig = "{:07d}.png".format(i)
                images_00.append(fig)
                
            row = math.ceil(len(images_00)/imagesPerRow)

            images = [Image.open(x) for x in images_00]
            widths, heights = zip(*(i.size for i in images))

            total_width = int(max(widths)*imagesPerRow)
            maxHeight = max(heights)
            max_height = int(maxHeight*row)

            new_im = Image.new('RGBA', (total_width, max_height), color="white")        # generate the new white background image for the new image

            x_offset = 0
            y_offset = 0
            count = 0
            for im in images:
                if count<imagesPerRow:
                    new_im.paste(im, (x_offset, y_offset*maxHeight))            # paste the selected images on the white background image
                    x_offset += im.size[0]
                    count += 1
                else:
                    count = 0
                    x_offset = 0
                    y_offset += 1

            new_im.save('../assets/viz_supermarket_{}.png'.format("01"))    # save the generated image in the folder assets

            os.chdir("..")

        cwd = os.getcwd()
        print("current directory: ", cwd)

    return None

#
# callback function to generate the visualization of heatmaps (within the time step range) for supermarket layout 02
#
@app.callback(
    Output('dummy5', 'children'),
    [Input('slider_supermarket_02', 'value'),
    Input('imageRow_02', 'value'),
    Input('viz_supermarket_02', 'submit_n_clicks')],
)
def vizSuperMarket02(value, rowValue, submit_n_clicks):
    print(value[0])
    print(value[1])
    
    if (submit_n_clicks > 0):
        cwd = os.getcwd()
        print("current directory: ", cwd)

        dir = os.path.join(cwd, "assets")

        #imageDir = ["01", "02"]
        if os.path.exists("02"):
            os.chdir("02")

            imdir = os.getcwd()
            print("current directory: ", imdir)

            imagesPerRow = rowValue             # number of images per row

            numImage = []
            for imageFile in glob.glob("**/*.png", recursive=True):
                numImage.append(imageFile)      # read all required images for generating the new image
            numOfImages = len(numImage)

            #if value[0] > numOfImages:
            #    value[0] = numOfImages-10
            #if value[1] > numOfImages:
            #    value[1] = numOfImages
            #if value[0] > value[1]:
            #    value[0] = numOfImages-10
            #    value[1] = numOfImages

            figureX = int(value[0] * (numOfImages-1))   # convert to the figure number
            figureY = int(value[1] * (numOfImages-1))   # convert to the figure number

            images_00 = []
            for i in range(figureX, figureY, 1):
                fig = "{:07d}.png".format(i)
                images_00.append(fig)
                
            row = math.ceil(len(images_00)/imagesPerRow)

            images = [Image.open(x) for x in images_00]
            widths, heights = zip(*(i.size for i in images))

            total_width = int(max(widths)*imagesPerRow)
            maxHeight = max(heights)
            max_height = int(maxHeight*row)

            new_im = Image.new('RGBA', (total_width, max_height), color="white")    # generate the new white background image for the new image

            x_offset = 0
            y_offset = 0
            count = 0
            for im in images:
                if count<imagesPerRow:
                    new_im.paste(im, (x_offset, y_offset*maxHeight))        # paste the selected images on the white background image
                    x_offset += im.size[0]
                    count += 1
                else:
                    count = 0
                    x_offset = 0
                    y_offset += 1

            new_im.save('../assets/viz_supermarket_{}.png'.format("02"))        # save the generated image in the folder assets

            os.chdir("..")

        cwd = os.getcwd()
        print("current directory: ", cwd)

    return None

#
# callback function to generate the summarized visualization of heatmaps for both supermarket layouts
#
@app.callback(
    Output('dummy6', 'children'),
    [Input('viz_II_supermarket', 'submit_n_clicks')],
)
def vizIISuperMarket(submit_n_clicks):
    if (submit_n_clicks > 0):
        numTxtFile0102 = []
        if os.path.exists("01"):
            os.chdir("01")
            integratedPlumes01 = np.loadtxt('integrated_plumes_store_data.dat', unpack = True)      # read the summurized heatmap value of supermarket 01
            os.chdir("..")
        
        if os.path.exists("02"):
            os.chdir("02")
            integratedPlumes02 = np.loadtxt('integrated_plumes_store_data.dat', unpack = True)      # read the summurized heatmap value of supermarket 02
            os.chdir("..")

        cm_data = [[0.2422, 0.1504, 0.6603],
        [0.2444, 0.1534, 0.6728],
        [0.2464, 0.1569, 0.6847],
        [0.2484, 0.1607, 0.6961],
        [0.2503, 0.1648, 0.7071],
        [0.2522, 0.1689, 0.7179],
        [0.254, 0.1732, 0.7286],
        [0.2558, 0.1773, 0.7393],
        [0.2576, 0.1814, 0.7501],
        [0.2594, 0.1854, 0.761],
        [0.2611, 0.1893, 0.7719],
        [0.2628, 0.1932, 0.7828],
        [0.2645, 0.1972, 0.7937],
        [0.2661, 0.2011, 0.8043],
        [0.2676, 0.2052, 0.8148],
        [0.2691, 0.2094, 0.8249],
        [0.2704, 0.2138, 0.8346],
        [0.2717, 0.2184, 0.8439],
        [0.2729, 0.2231, 0.8528],
        [0.274, 0.228, 0.8612],
        [0.2749, 0.233, 0.8692],
        [0.2758, 0.2382, 0.8767],
        [0.2766, 0.2435, 0.884],
        [0.2774, 0.2489, 0.8908],
        [0.2781, 0.2543, 0.8973],
        [0.2788, 0.2598, 0.9035],
        [0.2794, 0.2653, 0.9094],
        [0.2798, 0.2708, 0.915],
        [0.2802, 0.2764, 0.9204],
        [0.2806, 0.2819, 0.9255],
        [0.2809, 0.2875, 0.9305],
        [0.2811, 0.293, 0.9352],
        [0.2813, 0.2985, 0.9397],
        [0.2814, 0.304, 0.9441],
        [0.2814, 0.3095, 0.9483],
        [0.2813, 0.315, 0.9524],
        [0.2811, 0.3204, 0.9563],
        [0.2809, 0.3259, 0.96],
        [0.2807, 0.3313, 0.9636],
        [0.2803, 0.3367, 0.967],
        [0.2798, 0.3421, 0.9702],
        [0.2791, 0.3475, 0.9733],
        [0.2784, 0.3529, 0.9763],
        [0.2776, 0.3583, 0.9791],
        [0.2766, 0.3638, 0.9817],
        [0.2754, 0.3693, 0.984],
        [0.2741, 0.3748, 0.9862],
        [0.2726, 0.3804, 0.9881],
        [0.271, 0.386, 0.9898],
        [0.2691, 0.3916, 0.9912],
        [0.267, 0.3973, 0.9924],
        [0.2647, 0.403, 0.9935],
        [0.2621, 0.4088, 0.9946],
        [0.2591, 0.4145, 0.9955],
        [0.2556, 0.4203, 0.9965],
        [0.2517, 0.4261, 0.9974],
        [0.2473, 0.4319, 0.9983],
        [0.2424, 0.4378, 0.9991],
        [0.2369, 0.4437, 0.9996],
        [0.2311, 0.4497, 0.9995],
        [0.225, 0.4559, 0.9985],
        [0.2189, 0.462, 0.9968],
        [0.2128, 0.4682, 0.9948],
        [0.2066, 0.4743, 0.9926],
        [0.2006, 0.4803, 0.9906],
        [0.195, 0.4861, 0.9887],
        [0.1903, 0.4919, 0.9867],
        [0.1869, 0.4975, 0.9844],
        [0.1847, 0.503, 0.9819],
        [0.1831, 0.5084, 0.9793],
        [0.1818, 0.5138, 0.9766],
        [0.1806, 0.5191, 0.9738],
        [0.1795, 0.5244, 0.9709],
        [0.1785, 0.5296, 0.9677],
        [0.1778, 0.5349, 0.9641],
        [0.1773, 0.5401, 0.9602],
        [0.1768, 0.5452, 0.956],
        [0.1764, 0.5504, 0.9516],
        [0.1755, 0.5554, 0.9473],
        [0.174, 0.5605, 0.9432],
        [0.1716, 0.5655, 0.9393],
        [0.1686, 0.5705, 0.9357],
        [0.1649, 0.5755, 0.9323],
        [0.161, 0.5805, 0.9289],
        [0.1573, 0.5854, 0.9254],
        [0.154, 0.5902, 0.9218],
        [0.1513, 0.595, 0.9182],
        [0.1492, 0.5997, 0.9147],
        [0.1475, 0.6043, 0.9113],
        [0.1461, 0.6089, 0.908],
        [0.1446, 0.6135, 0.905],
        [0.1429, 0.618, 0.9022],
        [0.1408, 0.6226, 0.8998],
        [0.1383, 0.6272, 0.8975],
        [0.1354, 0.6317, 0.8953],
        [0.1321, 0.6363, 0.8932],
        [0.1288, 0.6408, 0.891],
        [0.1253, 0.6453, 0.8887],
        [0.1219, 0.6497, 0.8862],
        [0.1185, 0.6541, 0.8834],
        [0.1152, 0.6584, 0.8804],
        [0.1119, 0.6627, 0.877],
        [0.1085, 0.6669, 0.8734],
        [0.1048, 0.671, 0.8695],
        [0.1009, 0.675, 0.8653],
        [0.0964, 0.6789, 0.8609],
        [0.0914, 0.6828, 0.8562],
        [0.0855, 0.6865, 0.8513],
        [0.0789, 0.6902, 0.8462],
        [0.0713, 0.6938, 0.8409],
        [0.0628, 0.6972, 0.8355],
        [0.0535, 0.7006, 0.8299],
        [0.0433, 0.7039, 0.8242],
        [0.0328, 0.7071, 0.8183],
        [0.0234, 0.7103, 0.8124],
        [0.0155, 0.7133, 0.8064],
        [0.0091, 0.7163, 0.8003],
        [0.0046, 0.7192, 0.7941],
        [0.0019, 0.722, 0.7878],
        [0.0009, 0.7248, 0.7815],
        [0.0018, 0.7275, 0.7752],
        [0.0046, 0.7301, 0.7688],
        [0.0094, 0.7327, 0.7623],
        [0.0162, 0.7352, 0.7558],
        [0.0253, 0.7376, 0.7492],
        [0.0369, 0.74, 0.7426],
        [0.0504, 0.7423, 0.7359],
        [0.0638, 0.7446, 0.7292],
        [0.077, 0.7468, 0.7224],
        [0.0899, 0.7489, 0.7156],
        [0.1023, 0.751, 0.7088],
        [0.1141, 0.7531, 0.7019],
        [0.1252, 0.7552, 0.695],
        [0.1354, 0.7572, 0.6881],
        [0.1448, 0.7593, 0.6812],
        [0.1532, 0.7614, 0.6741],
        [0.1609, 0.7635, 0.6671],
        [0.1678, 0.7656, 0.6599],
        [0.1741, 0.7678, 0.6527],
        [0.1799, 0.7699, 0.6454],
        [0.1853, 0.7721, 0.6379],
        [0.1905, 0.7743, 0.6303],
        [0.1954, 0.7765, 0.6225],
        [0.2003, 0.7787, 0.6146],
        [0.2061, 0.7808, 0.6065],
        [0.2118, 0.7828, 0.5983],
        [0.2178, 0.7849, 0.5899],
        [0.2244, 0.7869, 0.5813],
        [0.2318, 0.7887, 0.5725],
        [0.2401, 0.7905, 0.5636],
        [0.2491, 0.7922, 0.5546],
        [0.2589, 0.7937, 0.5454],
        [0.2695, 0.7951, 0.536],
        [0.2809, 0.7964, 0.5266],
        [0.2929, 0.7975, 0.517],
        [0.3052, 0.7985, 0.5074],
        [0.3176, 0.7994, 0.4975],
        [0.3301, 0.8002, 0.4876],
        [0.3424, 0.8009, 0.4774],
        [0.3548, 0.8016, 0.4669],
        [0.3671, 0.8021, 0.4563],
        [0.3795, 0.8026, 0.4454],
        [0.3921, 0.8029, 0.4344],
        [0.405, 0.8031, 0.4233],
        [0.4184, 0.803, 0.4122],
        [0.4322, 0.8028, 0.4013],
        [0.4463, 0.8024, 0.3904],
        [0.4608, 0.8018, 0.3797],
        [0.4753, 0.8011, 0.3691],
        [0.4899, 0.8002, 0.3586],
        [0.5044, 0.7993, 0.348],
        [0.5187, 0.7982, 0.3374],
        [0.5329, 0.797, 0.3267],
        [0.547, 0.7957, 0.3159],
        [0.5609, 0.7943, 0.305],
        [0.5748, 0.7929, 0.2941],
        [0.5886, 0.7913, 0.2833],
        [0.6024, 0.7896, 0.2726],
        [0.6161, 0.7878, 0.2622],
        [0.6297, 0.7859, 0.2521],
        [0.6433, 0.7839, 0.2423],
        [0.6567, 0.7818, 0.2329],
        [0.6701, 0.7796, 0.2239],
        [0.6833, 0.7773, 0.2155],
        [0.6963, 0.775, 0.2075],
        [0.7091, 0.7727, 0.1998],
        [0.7218, 0.7703, 0.1924],
        [0.7344, 0.7679, 0.1852],
        [0.7468, 0.7654, 0.1782],
        [0.759, 0.7629, 0.1717],
        [0.771, 0.7604, 0.1658],
        [0.7829, 0.7579, 0.1608],
        [0.7945, 0.7554, 0.157],
        [0.806, 0.7529, 0.1546],
        [0.8172, 0.7505, 0.1535],
        [0.8281, 0.7481, 0.1536],
        [0.8389, 0.7457, 0.1546],
        [0.8495, 0.7435, 0.1564],
        [0.86, 0.7413, 0.1587],
        [0.8703, 0.7392, 0.1615],
        [0.8804, 0.7372, 0.165],
        [0.8903, 0.7353, 0.1695],
        [0.9, 0.7336, 0.1749],
        [0.9093, 0.7321, 0.1815],
        [0.9184, 0.7308, 0.189],
        [0.9272, 0.7298, 0.1973],
        [0.9357, 0.729, 0.2061],
        [0.944, 0.7285, 0.2151],
        [0.9523, 0.7284, 0.2237],
        [0.9606, 0.7285, 0.2312],
        [0.9689, 0.7292, 0.2373],
        [0.977, 0.7304, 0.2418],
        [0.9842, 0.733, 0.2446],
        [0.99, 0.7365, 0.2429],
        [0.9946, 0.7407, 0.2394],
        [0.9966, 0.7458, 0.2351],
        [0.9971, 0.7513, 0.2309],
        [0.9972, 0.7569, 0.2267],
        [0.9971, 0.7626, 0.2224],
        [0.9969, 0.7683, 0.2181],
        [0.9966, 0.774, 0.2138],
        [0.9962, 0.7798, 0.2095],
        [0.9957, 0.7856, 0.2053],
        [0.9949, 0.7915, 0.2012],
        [0.9938, 0.7974, 0.1974],
        [0.9923, 0.8034, 0.1939],
        [0.9906, 0.8095, 0.1906],
        [0.9885, 0.8156, 0.1875],
        [0.9861, 0.8218, 0.1846],
        [0.9835, 0.828, 0.1817],
        [0.9807, 0.8342, 0.1787],
        [0.9778, 0.8404, 0.1757],
        [0.9748, 0.8467, 0.1726],
        [0.972, 0.8529, 0.1695],
        [0.9694, 0.8591, 0.1665],
        [0.9671, 0.8654, 0.1636],
        [0.9651, 0.8716, 0.1608],
        [0.9634, 0.8778, 0.1582],
        [0.9619, 0.884, 0.1557],
        [0.9608, 0.8902, 0.1532],
        [0.9601, 0.8963, 0.1507],
        [0.9596, 0.9023, 0.148],
        [0.9595, 0.9084, 0.145],
        [0.9597, 0.9143, 0.1418],
        [0.9601, 0.9203, 0.1382],
        [0.9608, 0.9262, 0.1344],
        [0.9618, 0.932, 0.1304],
        [0.9629, 0.9379, 0.1261],
        [0.9642, 0.9437, 0.1216],
        [0.9657, 0.9494, 0.1168],
        [0.9674, 0.9552, 0.1116],
        [0.9692, 0.9609, 0.1061],
        [0.9711, 0.9667, 0.1001],
        [0.973, 0.9724, 0.0938],
        [0.9749, 0.9782, 0.0872],
        [0.9769, 0.9839, 0.0805]]
    
        #
        #   generate the summarized visualization of heatmap for supermarket layout 01
        #
        integratedPlumes01Flip = np.flip(integratedPlumes01,0)
        integratedPlumes01Log = (np.log10(integratedPlumes01Flip, out=np.zeros_like(integratedPlumes01Flip), where=(integratedPlumes01Flip > 0)))*18
        cmap = sns.cm.rocket_r
        plt.figure()
        #colormap = sns.color_palette("coolwarm", 600)
        #colormap = sns.color_palette("Reds", 600)
        colormap = LinearSegmentedColormap.from_list('parula', cm_data)
        ax_heatmap01 = sns.heatmap(integratedPlumes01Log, linewidth=0.0, cmap=colormap, xticklabels=False, yticklabels=False, cbar_kws={'label': '$\mathrm{Aerosols} / \mathrm{m}^3$'}, vmin=0.01, vmax=100)
        ax_heatmap01.figure.axes[-1].yaxis.label.set_size(18)
        cbar = ax_heatmap01.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        plt.title('Heatmap for supermarket 01', fontsize = 20)
        plt.savefig("assets/heatMap01.png")                     # generate and save the image in the folder assets

        #
        #   generate the summarized visualization of heatmap for supermarket layout 01
        #
        integratedPlumes02Flip = np.flip(integratedPlumes02,0)
        integratedPlumes02Log = (np.log10(integratedPlumes02Flip, out=np.zeros_like(integratedPlumes02Flip), where=(integratedPlumes02Flip > 0)))*18
        plt.figure()
        #colormap = sns.color_palette("coolwarm", 600)
        #colormap = sns.color_palette("Reds", 600)
        colormap = LinearSegmentedColormap.from_list('parula', cm_data)
        ax_heatmap02 = sns.heatmap(integratedPlumes02Log, linewidth=0.0, cmap=colormap, xticklabels=False, yticklabels=False, cbar_kws={'label': '$\mathrm{Aerosols} / \mathrm{m}^3$'}, vmin=0.01, vmax=100)
        ax_heatmap02.figure.axes[-1].yaxis.label.set_size(18)
        cbar = ax_heatmap02.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        plt.title('Heatmap for supermarket 02', fontsize = 20)
        plt.savefig("assets/heatMap02.png")                     # generate and save the image in the folder assets

#
# callback function to generate the visualization of lineplot and histogram
#
# lineplot: Number of customers inside the supermarket in every time step (the time steps have been normalized (between 0.0 and 1.0))
# Histogram: Number of customers exposed to different aerosol concentration levels
#
# Level 1: expose to between 1.0 and 5.0 aerosols/m^3. Low risk.
# Level 2: expose to between 5.0 and 10.0 aerosols/m^3
# Level 3: expose to between 10.0 and 50.0 aerosols/m^3
# Level 4: expose to >= 50.0 aerosols/m^3. High risk
#
@app.callback(
    Output('dummy7', 'children'),
    [Input('viz_III_supermarket', 'submit_n_clicks')],
)
def vizIIISuperMarket(submit_n_clicks):
    
    range_to_normalize = (0, 1)

    # explicit function to normalize array
    def normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    # get the number of steps of the results for each supermarket 01 and 02
    # the timestep is normalised such that the different time step for both supermarket can be plotted on the same lineplot
    if (submit_n_clicks > 0):
        steps0102 = []
        customer0102 = []
        if os.path.exists("01"):
            os.chdir("01")
            store = np.loadtxt('store_data.dat', unpack = True)
            steps0102.append(len(store[0]))
            customer = np.loadtxt('customer_data.dat', unpack = True)
            customer0102.append(len(customer[0]))
            os.chdir("..")
        if os.path.exists("02"):
            os.chdir("02")
            store = np.loadtxt('store_data.dat', unpack = True)
            steps0102.append(len(store[0]))
            customer = np.loadtxt('customer_data.dat', unpack = True)
            customer0102.append(len(customer[0]))
            os.chdir("..")
        xCustomer01 = np.arange(customer0102[0])
        xCustomer02 = np.arange(customer0102[1])
        xNormCustomer01 = normalize(np.arange(steps0102[0]), range_to_normalize[0], range_to_normalize[1])      # normalization of the time step for supermarket 01
        xNormCustomer02 = normalize(np.arange(steps0102[1]), range_to_normalize[0], range_to_normalize[1])      # normalization of the time step for supermarket 02

        # get the information of number of customers entered the plume level 1 to level 4 for supermarket 01 from the text file
        # level 1 means low risk
        # level 4 means high risk
        if os.path.exists("01"):
            os.chdir("01")
            storeData01 = np.loadtxt('store_data.dat', unpack = True)
            numCustomerInStore01 = storeData01[0]
            infectedCustomerInStore01 = storeData01[2]
            exposureDuringTimeStep01 = storeData01[3]

            customerData01 = np.loadtxt('customer_data.dat', unpack = True)
            infectedCustomer01 = customerData01[0][:customer0102[0]]    # infected customer before entering the supermarket
            exposureHist01 = customerData01[2][:customer0102[0]]
            timeSpent01 = customerData01[3][:customer0102[0]]
            exposureHistTime01 = customerData01[4][:customer0102[0]]
            exposureHistTimeThres01 = customerData01[5][:customer0102[0]]
            exposureHistTimeThresLevel101 = customerData01[6][:customer0102[0]]
            exposureHistTimeThresLevel201 = customerData01[7][:customer0102[0]]
            exposureHistTimeThresLevel301 = customerData01[8][:customer0102[0]]
            exposureHistTimeThresLevel401 = customerData01[9][:customer0102[0]]
            exposureHistTimeThresLevel501 = customerData01[10][:customer0102[0]]

            os.chdir("..")
        
        # get the information of number of customers entered the plume level 1 to level 4 for supermarket 02
        # level 1 means low risk
        # level 4 means high risk
        if os.path.exists("02"):
            os.chdir("02")
            storeData02 = np.loadtxt('store_data.dat', unpack = True)
            numCustomerInStore02 = storeData02[0]
            infectedCustomerInStore02 = storeData02[2]
            exposureDuringTimeStep02 = storeData02[3]

            customerData02 = np.loadtxt('customer_data.dat', unpack = True)
            infectedCustomer02 = customerData02[0][:customer0102[0]]    # infected customer before entering the supermarket
            exposureHist02 = customerData02[2][:customer0102[0]]
            timeSpent02 = customerData02[3][:customer0102[0]]
            exposureHistTime02 = customerData02[4][:customer0102[0]]
            exposureHistTimeThres02 = customerData02[5][:customer0102[0]]
            exposureHistTimeThresLevel102 = customerData02[6][:customer0102[0]]
            exposureHistTimeThresLevel202 = customerData02[7][:customer0102[0]]
            exposureHistTimeThresLevel302 = customerData02[8][:customer0102[0]]
            exposureHistTimeThresLevel402 = customerData02[9][:customer0102[0]]
            exposureHistTimeThresLevel502 = customerData02[10][:customer0102[0]]

            os.chdir("..")
        
        # plot the lineplot
        # lineplot: Number of customers inside the supermarket in every time step (the time steps have been normalized (between 0.0 and 1.0))
        plt.figure()
        #plt.figure(figsize=(80,30))
        ax_numCustomer = plt.axes()
        ax_numCustomer.plot(xNormCustomer01, numCustomerInStore01, label ='customers in supermarket 01')
        ax_numCustomer.plot(xNormCustomer02, numCustomerInStore02, label ='customers in supermarket 02')
        ax_numCustomer.plot(xNormCustomer01, infectedCustomerInStore01, label ='infected customers in supermarket 01')
        ax_numCustomer.plot(xNormCustomer02, infectedCustomerInStore02, label ='infected customers in supermarket 02')
        #legend = plt.legend(loc='center', bbox_to_anchor=(0.5,0.3), prop={'size': 13})
        legend = plt.legend(loc=2, prop={'size': 13})
        #legend.get_frame().set_faceco0lor('#FFFFFF')
        ax_numCustomer.set_xlim(0.0,1.0)
        ax_numCustomer.set_ylim(0,max(numCustomerInStore01)+80)
        plt.title('Number of customers and infected customers in supermarket 01 and 02', fontsize = 13)
        plt.xlabel('Normalized time step', fontsize = 12)
        plt.ylabel('Number of (infected) customers', fontsize = 12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig("assets/numCustomers.png")

        # get the information of number of customers entered the plume level 1 to level 4 for supermarket 01
        # level 1 means low risk
        # level 4 means high risk
        customerLevel501 = 0
        customerLevel401 = 0
        customerLevel301 = 0
        customerLevel201 = 0
        customerLevel101 = 0
        for i in xCustomer01:
            if (exposureHistTimeThresLevel101[i]>0 and exposureHistTimeThresLevel201[i]>0 and exposureHistTimeThresLevel301[i]>0 and exposureHistTimeThresLevel401[i]>0 and exposureHistTimeThresLevel501[i]>0):
                customerLevel501 += 1
            elif (exposureHistTimeThresLevel101[i]>0 and exposureHistTimeThresLevel201[i]>0 and exposureHistTimeThresLevel301[i]>0 and exposureHistTimeThresLevel401[i]>0):
                customerLevel401 += 1
            elif (exposureHistTimeThresLevel101[i]>0 and exposureHistTimeThresLevel201[i]>0 and exposureHistTimeThresLevel301[i]>0):
                customerLevel301 += 1
            elif (exposureHistTimeThresLevel101[i]>0 and exposureHistTimeThresLevel201[i]>0):
                customerLevel201 += 1
            elif (exposureHistTimeThresLevel101[i]>0):    
                customerLevel101 += 1
        customerLevels01 = [customerLevel201, customerLevel301, customerLevel401, customerLevel501]

        # get the information of number of customers entered the plume level 1 to level 4 for supermarket 02
        # level 1 means low risk
        # level 4 means high risk
        customerLevel502 = 0
        customerLevel402 = 0
        customerLevel302 = 0
        customerLevel202 = 0
        customerLevel102 = 0
        for i in xCustomer02:
            if (exposureHistTimeThresLevel102[i]>0 and exposureHistTimeThresLevel202[i]>0 and exposureHistTimeThresLevel302[i]>0 and exposureHistTimeThresLevel402[i]>0 and exposureHistTimeThresLevel502[i]>0):
                customerLevel502 += 1
            elif (exposureHistTimeThresLevel102[i]>0 and exposureHistTimeThresLevel202[i]>0 and exposureHistTimeThresLevel302[i]>0 and exposureHistTimeThresLevel402[i]>0):
                customerLevel402 += 1
            elif (exposureHistTimeThresLevel102[i]>0 and exposureHistTimeThresLevel202[i]>0 and exposureHistTimeThresLevel302[i]>0):
                customerLevel302 += 1
            elif (exposureHistTimeThresLevel102[i]>0 and exposureHistTimeThresLevel202[i]>0):
                customerLevel202 += 1
            elif (exposureHistTimeThresLevel102[i]>0):    
                customerLevel102 += 1
        customerLevels02 = [customerLevel202, customerLevel302, customerLevel402, customerLevel502]

        #
        # plot the histogram: number of customers exposed to different aerosol concentration levels
        # level 1 means low risk
        # level 4 means high risk
        #
        levels = ['level 1', 'level 2', 'level 3', 'level 4']
        ind = np.arange(len(levels))
        plt.figure()
        widthBar = 0.3
        plt.bar(ind, customerLevels01, widthBar, label='supermarket 01')
        plt.bar(ind+widthBar, customerLevels02, widthBar, label='supermarket 02')
        plt.title('Number of customers in each level in supermarket 01 and 02', fontsize = 13)
        plt.xlabel('Levels', fontsize = 12)
        plt.ylabel('Number of customers', fontsize = 12)
        plt.xticks(ind+widthBar/2, levels)
        legend = plt.legend(loc='best', prop={'size': 13})
        legend.get_frame().set_facecolor('0.90')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig("assets/numCustomersLevels.png")        # plot the histogram and save the image in folder assets

# Driver program
if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False)
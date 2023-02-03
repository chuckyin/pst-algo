#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:32:20 2023

@author: melshaer0612
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.interpolate import griddata


def process_csv(filename):
    df = pd.read_csv(filename)
    df_sub = df[['x', 'y', 'xi', 'yi', 'frame_num']]
    df_sub = df_sub.rename(columns={'x':'x_temp', 'y':'y_temp'})

    df['xi1'] = df['xi'] - 1
    df['yi1'] = df['yi'] - 1

    xi = df_sub['xi'].values
    yi = df_sub['yi'].values
    frame_num = df_sub['frame_num'].values
    v = np.vstack([xi, yi, frame_num]).T.tolist()
    vt = [tuple(vi) for vi in v]
    df_sub['key'] = vt

    xi1 = df['xi1'].values
    yi1 = df['yi1'].values
    xi_df = df['xi'].values
    yi_df = df['yi'].values
    frame_num_df = df['frame_num'].values
    vx_df = np.vstack([xi1, yi_df, frame_num_df]).T.tolist()
    vy_df = np.vstack([xi_df, yi1, frame_num_df]).T.tolist()
    vxt_df = [tuple(vi) for vi in vx_df]
    vyt_df = [tuple(vi) for vi in vy_df]
    df['keyx'] = vxt_df
    df['keyy'] = vyt_df

    merged_df_x = pd.merge(df, df_sub[['x_temp', 'key']], left_on=['keyx'], right_on=['key'], how='left').drop(columns=['keyx', 'key'])
    merged_df = pd.merge(merged_df_x, df_sub[['y_temp', 'key']], left_on=['keyy'], right_on=['key'], how='left').drop(columns=['keyy', 'key'])

    merged_df['dx'] = merged_df['x'] - merged_df['x_temp']
    merged_df['dy'] = merged_df['y'] - merged_df['y_temp']

    df2 = pd.DataFrame()
    df2['dx_med'] = merged_df.groupby(['xi', 'yi'])['dx'].median()
    df2['dy_med'] = merged_df.groupby(['xi', 'yi'])['dy'].median()

    df2.reset_index(inplace=True)

    v_df2 = np.vstack([df2['xi'].values, df2['yi'].values]).T.tolist()
    v_merged = np.vstack([merged_df['xi'].values, merged_df['yi'].values]).T.tolist()
    vt_df2 = [tuple(vi) for vi in v_df2]
    vt_merged = [tuple(vi) for vi in v_merged]
    df2['key'] = vt_df2
    merged_df['key'] = vt_merged

    merged_df2 = pd.merge(merged_df, df2[['dx_med', 'dy_med', 'key']], on=['key'], how='left').drop(columns=['key'])

    merged_df2['dx_norm'] = merged_df2['dx'] / merged_df2['dx_med']
    merged_df2['dy_norm'] = merged_df2['dy'] / merged_df2['dy_med']
    
    # ------------------------------?
    # Plots
    y = merged_df2['yi'].tolist()
    x = merged_df2['xi'].tolist()
    dx = merged_df2['dx_norm'].tolist()
    dy = merged_df2['dy_norm'].tolist()
    
    # Suggestion 1: Hexbin Plots
    plt.hexbin(x, y, dx, cmap=cm.jet)
    plt.colorbar()
    plt.title('dx')
    plt.figure()
    plt.hexbin(x, y, dy, cmap=cm.jet)
    plt.colorbar()
    plt.title('dy')

    # Suggestion 2: Scatter Plots
    plt.scatter(x, y, c=dx, cmap=cm.jet)
    plt.colorbar()
    plt.figure()
    plt.scatter(x, y, c=dy, cmap=cm.jet)
    plt.colorbar()
    
    # Suggestion 3: Joint Distribution Plots
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "hex", data = merged_df2['dx'])
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "kde", data = merged_df2['dx'])
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "scatter", data = merged_df2['dx'])
    
    # Suggestion 4: Contour Plots
    data = np.array([x, y, dx]).T
    X, Y = np.meshgrid(data[:,0], data[:,1])
    Z = griddata((data[:,0], data[:,1]), data[:,2], (X, Y), method='nearest')
    plt.contourf(X, Y, Z)
    
    # Or
    levels = 0.5
    plt.contour(X, Y, Z, levels=levels)
    
    # Or
    merged_na = merged_df2.dropna()
    y = merged_na['yi'].tolist()
    x = merged_na['xi'].tolist()
    dx = merged_na['dx_norm'].tolist()
    dy = merged_na['dy_norm'].tolist()
    plt.tricontourf(x, y, dx)
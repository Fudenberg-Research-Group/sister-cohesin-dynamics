
def useLargeSize(plt, axis, marker_lines = None, fontsize = 'xx-large', fontproperties=None, LW=2.3):
    '''
      将X,Y坐标轴的标签、刻度以及legend都使用大字体,
      所有线条采用粗线
    '''
    axis.xaxis.get_label().set_size(fontsize)
    axis.yaxis.get_label().set_size(fontsize)
    axis.yaxis.set_ticks_position('both')
    axis.xaxis.set_ticks_position('both')
    axis.tick_params(labeltop=False, labelright=False)
    #分别设置x轴和y轴上刻度值的字体大小
    for label in axis.xaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in axis.yaxis.get_ticklabels():
        label.set_fontsize(fontsize) 
    # 设置线的粗细
    # LW = 2.3
    axis.xaxis.set_tick_params(width=LW, direction='in')
    axis.yaxis.set_tick_params(width=LW, direction='in')
    for ax in ['top','bottom','left','right']:
        axis.spines[ax].set_linewidth(LW)
    for line in axis.get_lines():
        line.set_lw( LW )
    leg = axis.get_legend()
    if(leg):
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        if(fontproperties):
            plt.setp(ltext, fontproperties=fontproperties)        
        plt.setp(ltext, fontsize='x-large')
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        plt.setp(llines,linewidth= LW )
        if(marker_lines and len(marker_lines)>=len(llines)):
            for i in range(0,len(llines)):
                plt.setp(llines[i], 
                    marker = marker_lines[i].get_marker(), 
                    markeredgecolor= marker_lines[i].get_markeredgecolor(),\
                    markerfacecolor= marker_lines[i].get_markerfacecolor(),\
                    markeredgewidth= marker_lines[i].get_markeredgewidth(),
                    markersize= marker_lines[i].get_markersize() )
    plt.tight_layout()



def useLargeSize1(plt, axis, marker_lines=None, fontsize='xx-large', fontname=None, fontproperties=None, LW=2.3,\
                   axis_fontsize=None, legend_fontsize=None):
    '''
    将X,Y坐标轴的标签、刻度以及legend都使用大字体,
    所有线条采用粗线
    '''
    if fontname:
        plt.rcParams['font.family'] = fontname

    # Set X and Y axis label size
    if axis_fontsize is not None:
        axis.xaxis.label.set_size(axis_fontsize)
        axis.yaxis.label.set_size(axis_fontsize)

    # Set X and Y axis tick label size
    for label in axis.xaxis.get_ticklabels():
        if fontsize is not None:
            label.set_fontsize(fontsize)

    for label in axis.yaxis.get_ticklabels():
        if fontsize is not None:
            label.set_fontsize(fontsize)

    # Set X and Y axis tick width and direction
    axis.xaxis.set_tick_params(width=LW, direction='in')
    axis.yaxis.set_tick_params(width=LW, direction='in')
    # Set line width
    for ax in ['top', 'bottom', 'left', 'right']:
        axis.spines[ax].set_linewidth(LW)
    for line in axis.get_lines():
        line.set_lw(LW)

    # Set legend size
    leg = axis.get_legend()
    if (leg):
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        if (fontproperties):
            plt.setp(ltext, fontproperties=fontproperties)
        if legend_fontsize is not None:
            plt.setp(ltext, fontsize=legend_fontsize)
        else:
            plt.setp(ltext, fontsize='x-large')
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        plt.setp(llines, linewidth=LW)
        if (marker_lines and len(marker_lines) >= len(llines)):
            for i in range(0, len(llines)):
                plt.setp(llines[i],
                         marker=marker_lines[i].get_marker(),
                         markeredgecolor=marker_lines[i].get_markeredgecolor(),
                         markerfacecolor=marker_lines[i].get_markerfacecolor(),
                         markeredgewidth=marker_lines[i].get_markeredgewidth(),
                         markersize=marker_lines[i].get_markersize())
    plt.tight_layout()



def useLargeSize2(plt, axis, axis2=None, marker_lines = None, fontsize = 'xx-large', fontproperties=None, LW=2.3):     
    '''       
    将X,Y坐标轴的标签、刻度以及legend都使用大字体,       
    所有线条采用粗线     
    '''     
    # Apply formatting to the primary axis
    axis.xaxis.get_label().set_size(fontsize)     
    axis.yaxis.get_label().set_size(fontsize)     
    axis.yaxis.set_ticks_position('both')     
    axis.xaxis.set_ticks_position('both')     
    axis.tick_params(labeltop=False, labelright=False)     
    
    #分别设置x轴和y轴上刻度值的字体大小     
    for label in axis.xaxis.get_ticklabels():         
        label.set_fontsize(fontsize)     
    for label in axis.yaxis.get_ticklabels():         
        label.set_fontsize(fontsize)      

    # 设置线的粗细     
    axis.xaxis.set_tick_params(width=LW, direction='in')     
    axis.yaxis.set_tick_params(width=LW, direction='in')     
    for ax in ['top','bottom','left','right']:         
        axis.spines[ax].set_linewidth(LW)     
    for line in axis.get_lines():         
        line.set_lw( LW )     
    
    # Apply formatting to the secondary axis (ax2) if provided
    if axis2 is not None:
        axis2.yaxis.get_label().set_size(fontsize)
        axis2.tick_params(labelright=True)  # Show right labels for ax2
        
        # Set font size for right y-axis tick labels
        for label in axis2.yaxis.get_ticklabels():
            label.set_fontsize(fontsize)
        
        # Set line width for right y-axis
        axis2.yaxis.set_tick_params(width=LW, direction='in')
        
        # Set spine width for right axis
        axis2.spines['right'].set_linewidth(LW)
    
    # Handle legend (only for primary axis to avoid duplicates)
    leg = axis.get_legend()     
    if(leg):         
        ltext  = leg.get_texts()  # all the text.Text instance in the legend         
        if(fontproperties):             
            plt.setp(ltext, fontproperties=fontproperties)                 
            plt.setp(ltext, fontsize='x-large')         
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend         
        plt.setp(llines,linewidth= LW )         
        if(marker_lines and len(marker_lines)>=len(llines)):             
            for i in range(0,len(llines)):                 
                plt.setp(llines[i],                      
                    marker = marker_lines[i].get_marker(),                      
                    markeredgecolor= marker_lines[i].get_markeredgecolor(),
                    markerfacecolor= marker_lines[i].get_markerfacecolor(),
                    markeredgewidth= marker_lines[i].get_markeredgewidth(),                     
                    markersize= marker_lines[i].get_markersize() )     
    
    plt.tight_layout()
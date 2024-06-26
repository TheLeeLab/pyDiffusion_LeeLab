#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:02:24 2023

Class related to making figure-quality plots

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/
 
The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should 
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font

@author: jbeckwith
"""
import matplotlib  # requires 3.8.0
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, poster=False):
        self.poster = poster
        self = self
        return

    def two_column_plot(
        self, nrows=1, ncolumns=1, heightratio=[1], widthratio=[1], height=0
    ):
        """two_column_plot function
        takes data and makes a two-column width figure

        Args:
            nrows (int): number of rows
            ncolumns (int): number of columns
            heightratio (np.1darray): array of same length as number of rows
            widthratio (np.1darray): array of same length as number of columns
            height (float): if set, figure will be this height

        Returns:
            fig (figure object): figure
            axs (axes object): axes
        """

        # first, check everything matches
        try:
            if len(heightratio) != nrows:
                raise Exception("Number of height ratios incorrect")
            if len(widthratio) != ncolumns:
                raise Exception("Number of width ratios incorrect")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 1

        xsize = 6.69  # 3.33 inches for one-column figure
        if height == 0:
            ysize = np.min([3.5 * nrows, 8.25])  # maximum size in y can be 8.25
        else:
            ysize = height

        plt.rcParams["figure.figsize"] = [xsize, ysize]
        plt.rcParams["font.size"] = fontsz
        plt.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.linewidth"] = lw  # set the value globally

        fig, axs = plt.subplots(
            nrows, ncolumns, height_ratios=heightratio, width_ratios=widthratio
        )  # create number of panels

        # clean up axes, tick parameters
        if nrows * ncolumns == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw * 4)
            axs.yaxis.set_tick_params(width=lw, length=lw * 4)
            axs.tick_params(axis="both", pad=1.2)
        elif nrows * ncolumns == 2:
            for i in np.arange(2):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        elif nrows * ncolumns == len(widthratio):
            for i in np.arange(len(widthratio)):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        else:
            for i in np.arange(nrows):
                for j in np.arange(ncolumns):
                    axs[i, j].xaxis.set_tick_params(width=0.5, length=lw * 4)
                    axs[i, j].yaxis.set_tick_params(width=0.5, length=lw * 4)
                    axs[i, j].tick_params(axis="both", pad=1.2)
        return fig, axs

    def multi3Dtrack_plot(
        self,
        axs,
        coordinates_dict,
        lw=0.75,
        label="",
        xaxislabel="x axis",
        yaxislabel="y axis",
        zaxislabel="z axis",
        ls="-",
    ):

        for key in coordinates_dict.keys():
            data = coordinates_dict[key]
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            axs.plot(x, y, z, lw=lw, ls=ls)

        axs.set(xlabel=xaxislabel, ylabel=yaxislabel, zlabel=zaxislabel)
        return axs

    def one_column_plot(self, npanels=1, ratios=[1], height=None, threedim=False):
        """one_column_plot function
        takes data and makes a one-column width figure

        Args:
            npanels (int): number of panels in the figure
            ratios (np.1darray): size ratios of panels
            height (float): figure height
            threedim (boolean): if True, make a 3D projection figure

        Returns:
            fig (figure object): figure
            axs (axes object): axes
        """

        # first, check everything matches
        try:
            if len(ratios) != npanels:
                raise Exception("Number of ratios incorrect")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 0.5

        xsize = 3.33  # 3.33 inches for one-column figure
        if height is not None:
            ysize = np.min([height, 8.25])  # maximum size in y can be 8.25
        else:
            ysize = np.min([3.5 * npanels, 8.25])  # maximum size in y can be 8.25

        plt.rcParams["figure.figsize"] = [xsize, ysize]
        plt.rcParams["font.size"] = fontsz
        plt.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.linewidth"] = lw  # set the value globally

        if threedim == True:
            fig, axs = plt.subplots(
                npanels, 1, height_ratios=ratios, subplot_kw=dict(projection="3d")
            )  # create number of panels
        else:
            fig, axs = plt.subplots(
                npanels, 1, height_ratios=ratios
            )  # create number of panels

        # clean up axes, tick parameters
        if npanels == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw * 4)
            axs.yaxis.set_tick_params(width=lw, length=lw * 4)
            axs.tick_params(axis="both", pad=1.2)
        else:
            for i in np.arange(npanels):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        return fig, axs

    def line_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        color="k",
        lw=0.75,
        label="",
        xaxislabel="x axis",
        yaxislabel="y axis",
        ls="-",
    ):
        """line_plot function
        takes data and makes a line plot

        Args:
            x (np.1darray): x data
            y (np.1darray): y data
            xlim is x limits; default is None (which computes max/min)
            ylim is y limits; default is None (which computes max/min)
            color is line colour; default is black
            lw is line width (default 0.75)
            label is label; default is nothing
            xaxislabel is x axis label (default is 'x axis')
            yaxislabel is y axis label (default is 'y axis')

        Returns:
            axs is axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(x), np.max(x)])
        if ylim is None:
            ylim = np.array([np.min(y), np.max(y)])
        axs.plot(x, y, lw=lw, color=color, label=label, ls=ls)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def histogram_plot(
        self,
        axs,
        data,
        bins,
        xlim=None,
        ylim=None,
        histcolor="gray",
        xaxislabel="x axis",
        alpha=1,
        histtype="bar",
        density=True,
    ):
        """histogram_plot function
        takes data and makes a histogram

        Args:
            data (np.1darray): data array
            bins (np.1darray): bin array
            xlim (boolean or list of two floats): default is None (which computes min/max of x), otherwise provide a min/max
            ylim (boolean or list of two floats): default is None (which computes min/max of y), otherwise provide a min/max
            histcolor (string): histogram colour (default is gray)
            xaxislabel (string): x axis label (default is 'x axis')
            alpha (float): histogram transparency (default 1)
            histtype (string): histogram type, default bar
            density (boolean): if to plot as pdf, default True

        Returns:
            axs (axis): axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(data), np.max(data)])

        axs.hist(
            data,
            bins=bins,
            density=density,
            color=histcolor,
            alpha=alpha,
            histtype=histtype,
        )
        axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        if density == True:
            axs.set_ylabel("probability density", fontsize=fontsz)
        else:
            axs.set_ylabel("frequency", fontsize=fontsz)
        axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        return axs

    def scatter_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        label="",
        edgecolor="k",
        facecolor="white",
        s=5,
        lw=0.75,
        xaxislabel="x axis",
        yaxislabel="y axis",
        alpha=1,
    ):
        """scatter_plot function
        takes scatter data and makes an scatter plot

        Args:
            xdata (np.1darray): scatter points, x
            ydata (np.1darray): scatter points, y
            xlim (np.1darray): x limits
            ylim (np.1darray): y limits
            label (string): any annotation
            edgecolor (string): colour of scatter plot edges
            facecolor (string): colour of scatter point faces
            alpha (float): transparency of points

        Returns:
            axs (axis): axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(x), np.max(x)])
        if ylim is None:
            ylim = np.array([np.min(y), np.max(y)])
        axs.scatter(
            x,
            y,
            s=s,
            edgecolors=edgecolor,
            facecolor=facecolor,
            lw=lw,
            label=label,
            alpha=alpha,
        )
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def kusumi_plot(
        self,
        axs,
        deltaT,
        MSD,
        xlim=None,
        ylim=None,
        label="",
        edgecolor="k",
        facecolor="white",
        alpha=1,
        s=5,
        lw=0.75,
        xaxislabel=r"$\Delta$t",
        yaxislabel=r"MSD/4$\Delta$t",
    ):
        """kusumi_plot function
        takes MSD and deltaT and makes a kusumi plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            deltaT (np.1darray): deltaT array
            MSD (np.1darray): MSD array
            xlim (np.1darray): x limits
            ylim (np.1darray): y limits
            label (str): label of data
            edgecolor (str): colour of edges (default black)
            facecolor (str): colour of face (default white)
            alpha (float): transparency of scatter point (default 1)
            s (float): size of data point
            lw (float): line with (default 0.75)
            xaxislabel (string): x axis label (default $\Delta$t)
            yaxislabel (string): x axis label (default MSD/4$\Delta$t)

        Returns:
            axs (axis): axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8
        y = MSD / (4 * deltaT)
        if xlim is None:
            xlim = np.array([np.min(deltaT), np.max(deltaT)])
        if ylim is None:
            ylim = np.array([np.min(y) / 10, np.max(y) * 10])

        axs.scatter(
            deltaT,
            y,
            s=s,
            edgecolors=edgecolor,
            facecolor=facecolor,
            lw=lw,
            label=label,
            alpha=alpha,
        )
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)

        axs.set_yscale("log")

        return axs

    def image_plot(
        self,
        axs,
        data,
        vmin=None,
        vmax=None,
        cmap="binary",
        cbar="on",
        cbarlabel="intensity",
        label="",
        labelcolor="black",
        pixelsize=69,
        scalebarsize=5000,
        scalebarlabel=r"5$\,\mu$m",
        alpha=1,
    ):
        """image_plot function
        takes image data and makes an image plot

        Args:
            data (np.2darray): image
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""
        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )
        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
            cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=8)
            cbar.ax.tick_params(labelsize=7, pad=0.1, width=0.5, length=2)
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize / pixelsize
        scalebar = AnchoredSizeBar(
            axs.transData,
            pixvals,
            scalebarlabel,
            "lower right",
            pad=0.1,
            color=labelcolor,
            frameon=False,
            size_vertical=1,
        )

        axs.add_artist(scalebar)
        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=6,
        )

        return axs

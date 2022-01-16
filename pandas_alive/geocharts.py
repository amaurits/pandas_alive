""" Implementation of geoplots with Geopandas
"""

import datetime
import typing
from typing import Mapping

import attr
import geopandas

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
import numpy as np
import pandas as pd

from matplotlib import colors, ticker, transforms
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap

from ._base_chart import _BaseChart


@attr.s
class MapChart(_BaseChart):
    """
    Map chart using Geopandas

    Args:
        _BaseChart ([type]): Base chart for all chart classes

    Raises:
        ValueError: [description]
    """

    basemap_format: typing.Dict = attr.ib()
    enable_markersize: bool = attr.ib()
    scale_markersize: float = attr.ib()
    use_dfcolors: bool = attr.ib()

    def __attrs_post_init__(self):
        """ Properties to be determined after initialization
        """
        self.df = self.df.copy()
        try:
            import descartes
        except:
            raise ModuleNotFoundError(
                "Ensure to install `descartes` if using geopandas with pandas_alive"
            )

        from shapely.geometry import Point

        for shape in self.df.geometry:
            if type(shape) == Point:
                self.enable_markersize = True
                break

        if self.df.crs != "EPSG:3857" and self.basemap_format:
            self.df = self.df.to_crs(3857)

        # Convert all columns except geometry to datetime
        try:
            self.df = self.convert_data_cols_to_datetime(self.df)
            # if self.interpolate_period and not self.use_dfcolors:
            self.df = self.get_interpolated_geo_df(self.df)
        except:
            import warnings

            warnings.warn(
                "Pandas_Alive failed to convert columns to datetime, setting interpolate_period to False and retrying..."
            )
            self.interpolate_period = False
            self.df = self.get_interpolated_geo_df(self.df)

        temp_gdf = self.df.copy()
        self.df = pd.DataFrame(self.df)
        self.df = self.df.drop("geometry", axis=1)

        # if self.fig is None:
        #     self.fig, self.ax = self.create_figure()
        #     self.figsize = self.fig.get_size_inches()
        # else:
        #     self.fig = plt.figure()
        #     self.ax = plt.axes()
        if self.fig is None:
            self.fig, self.ax = self.create_figure()
            self.figsize = self.fig.get_size_inches()
        else:
            # This will use `fig=` input by user and gets its first axis
            self.ax = self.fig.get_axes()[0]
            self.ax.tick_params(labelsize=self.tick_label_size)
        self.fig.set_tight_layout(False)
        if self.title:
            self.ax.set_title(self.title)
        if self.enable_progress_bar:
            self.setup_progress_bar()

        self.df = temp_gdf

    def create_figure(self) -> typing.Tuple[plt.figure, plt.axes]:
        """ Create base figure with styling, can be overridden if styling unwanted

        Returns:
            typing.Tuple[plt.figure,plt.figure.axes]: Returns Figure instance and the axes initialized within
        """

        fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)
        # limit = (0.2, self.n_bars + 0.8)
        # rect = self.calculate_new_figsize(fig)
        rect = [0, 0, 1, 1]  # left, bottom, width, height
        ax = fig.add_axes(rect)
        ax = self.apply_style(ax)

        return fig, ax

    def get_data_cols(self, gdf: geopandas.GeoDataFrame) -> typing.List:
        """
        Get data columns from GeoDataFrame (this excludes geometry)

        Args:
            gdf (geopandas.GeoDataFrame): Input GeoDataframe

        Returns:
            typing.List: List of columns except geometry
        """
        return gdf.loc[:, gdf.columns != "geometry"].columns.tolist()

    def convert_data_cols_to_datetime(
        self, gdf: geopandas.GeoDataFrame
    ) -> geopandas.GeoDataFrame:
        """
        Convert all data columns to datetime with `pd.to_datetime`

        Args:
            gdf (geopandas.GeoDataFrame): Input GeoDataFrame

        Returns:
            geopandas.GeoDataFrame: GeoDataFrame with data columns converted to `Timestamp`
        """
        converted_column_names = []
        for col in gdf.columns:
            if col != "geometry":
                col = pd.to_datetime(col)

            converted_column_names.append(col)
        gdf.columns = converted_column_names
        return gdf

    def get_interpolated_geo_df(
        self, gdf: geopandas.GeoDataFrame
    ) -> geopandas.GeoDataFrame:
        """
        Interpolates GeoDataFrame by splitting data from geometry, interpolating and joining back together

        Args:
            gdf (geopandas.GeoDataFrame): Input GeoDataFrame

        Returns:
            geopandas.GeoDataFrame: Interpolated GeoDataFrame
        """

        # Separate data from geometry
        temp_df = pd.DataFrame(gdf)
        temp_df = temp_df.drop("geometry", axis=1)
        temp_df = temp_df.T
        geometry_column = gdf.geometry

        # Interpolate data (previously used to call super().get_interpolated_df, without use_dfcolors parameter)
        interpolated_df = self.get_interpolated_df(
            temp_df, self.steps_per_period, self.interpolate_period, self.use_dfcolors
        )

        # Rejoin data with geometry
        interpolated_df = interpolated_df.T
        interpolated_df["geometry"] = geometry_column

        return geopandas.GeoDataFrame(interpolated_df)

    def get_interpolated_df(
            self, df: pd.DataFrame, steps_per_period: int, interpolate_period: bool, use_dfcolors=False,
    ) -> pd.DataFrame:
        """ Get interpolated dataframe to span total animation

        Supersedes the same function in _base_chart.py

        Args:
            df (pd.DataFrame): Input dataframe
            steps_per_period (int): The number of steps to go from one period to the next. Data will show linearly between each period
            interpolate_period (bool): Whether to interpolate the period, must be datetime index
            use_dfcolors (bool): Whether data to plot are hex colors rather than value

        Returns:
            pd.DataFrame: Interpolated dataframe
        """

        # Period interpolated to match other charts for multiple plotting
        # https://stackoverflow.com/questions/52701330/pandas-reindex-and-interpolate-time-series-efficiently-reindex-drops-data

        # First generate new rows, and interpolate the index values
        # (This code is identical to the first part of this function in _base_chart.py)
        interpolated_df = df.reset_index()
        interpolated_df.index = interpolated_df.index * steps_per_period
        new_index = range(interpolated_df.index[-1] + 1)
        interpolated_df = interpolated_df.reindex(new_index)
        if interpolate_period:
            if interpolated_df.iloc[:, 0].dtype.kind == "M":
                first, last = interpolated_df.iloc[[0, -1], 0]
                dr = pd.date_range(first, last, periods=len(interpolated_df.index))
                interpolated_df.iloc[:, 0] = dr
            else:
                interpolated_df.iloc[:, 0] = interpolated_df.iloc[:, 0].interpolate()
        else:
            interpolated_df.iloc[:, 0] = interpolated_df.iloc[:, 0].fillna(
                method="ffill"
            )

        interpolated_df = interpolated_df.set_index(interpolated_df.columns[0])

        # Now interpolate the actual data
        if self.use_dfcolors:  # This must be a df of hex colors, not normal values
            # Note that we're not doing anything special to handle time as an index
            # since basic subtraction and division operations will work correctly as is.
            interpolated_df = interpolate_hexcolor_df(interpolated_df)
        else:  # This is the same as the _base_chart.py code again
            if interpolate_period and isinstance(self.df.index, pd.DatetimeIndex):
                interpolated_df = interpolated_df.interpolate(method = "time")
            else:
                interpolated_df = interpolated_df.interpolate()

        return interpolated_df

    def plot_geo_data(self, i: int, gdf: geopandas.GeoDataFrame) -> None:
        """
        Plot GeoDataFrame using the plot accessor from Geopandas

        https://geopandas.org/reference.html#geopandas.GeoDataFrame.plot

        Args:
            i (int): Frame to plot
            gdf (geopandas.GeoDataFrame): Source GeoDataFrame
        """
        # fig, ax = plt.subplots(figsize=(5,3), dpi=100)
        # self.ax.clear()
        if self.use_dfcolors:
            gdf.plot(
                color=gdf[gdf.columns[i]],
                ax=self.ax,
                markersize=None,
                **self.kwargs,
            )
        else:
            column_to_plot = gdf.columns[i]
            gdf.plot(
                column=column_to_plot,
                ax=self.ax,
                markersize=gdf[column_to_plot] * self.scale_markersize if self.enable_markersize else None,
                cmap=self.cmap,
                **self.kwargs,
            )

        if self.basemap_format:
            try:
                import contextily

                if isinstance(self.basemap_format, dict):
                    contextily.add_basemap(self.ax, **self.basemap_format)
                else:
                    contextily.add_basemap(self.ax)

            except ImportError:

                raise ModuleNotFoundError(
                    "Ensure contextily is installed for basemap functionality https://github.com/geopandas/contextily"
                )

        return self.ax

    def anim_func(self, i: int) -> None:
        """ Animation function

        Args:
            i (int): Index of frame of animation
        """
        if self.enable_progress_bar:
            self.update_progress_bar()

        self.ax.clear()
        self.ax.set_axis_off()
        self.plot_geo_data(i, self.df)
        if self.period_fmt:
            self.show_period(i)

    def init_func(self) -> None:
        """ Initialization function for animation
        """
        column_to_plot = self.df.columns[0]
        self.df.plot(
            column=column_to_plot,
            markersize=self.df[column_to_plot],
            # cmap='viridis',
        )
        # self.ax.scatter([], [])

    def get_frames(self):
        """
        Get number of frames to animate
        """
        return range(len(self.get_data_cols(self.df)))

    def show_period(self, i: int) -> None:
        """
        Show period label on plot

        Args:
            i (int): Frame number of animation to take slice of DataFrame and retrieve current index for show as period

        Raises:
            ValueError: If custom period label location is used must contain `x`, `y` and `s` in dictionary.
        """
        if self.period_label:
            if self.period_fmt:
                idx_val = self.df.columns[i]
                if type(idx_val) == pd.Timestamp:  # Date time
                    s = idx_val.strftime(self.period_fmt)
                else:
                    s = self.period_fmt.format(x=idx_val)
            else:
                s = self.df.columns.astype(str)[i]
            num_texts = len(self.ax.texts)
            if num_texts == 0:
                # first frame
                self.ax.text(
                    s=s,
                    transform=self.ax.transAxes,
                    **self.get_period_label(self.period_label),
                )
            else:
                self.ax.texts[0].set_text(s)


def colorFader_array(c1, c2, mix=0):
    """Fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1).

    See Marcus Dutschke answer here:
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    """
    return np.array([colors.to_hex((1-mix_i) * np.array(colors.to_rgb(c1_i)) \
                                       + mix_i * np.array(colors.to_rgb(c2_i))) \
                     for c1_i, c2_i, mix_i in zip(c1, c2, mix)])


def interpolate_hexcolor_df(df):
    """Take a dataframe with columns that are hex color values, possibly with missing values inbetween.

    Interpolate by "averaging" the colors over the missing range.

    Inspired by the answer by jdehesa at
    https://stackoverflow.com/questions/41895857/creating-a-custom-interpolation-function-for-pandas

    Used in color interpolation for a dynamic chart, in pandas_alive.
    """
    # Extract into numpy array
    vals = df.values.copy()

    # Produce a mask of the elements that are NaN
    empty = np.any(pd.isnull(vals), axis=1)

    # Positions of the valid values
    valid_loc = np.argwhere(~empty).squeeze(axis=-1)

    # Indices (e.g. time) of the valid values
    valid_index = df.index[valid_loc].values

    # Positions of the missing values
    empty_loc = np.argwhere(empty).squeeze(axis=-1)

    # Discard missing values before first or after last valid
    empty_loc = empty_loc[(empty_loc > valid_loc.min()) & (empty_loc < valid_loc.max())]

    # Index value for missing values
    empty_index = df.index[empty_loc].values

    # Get valid values to use as interpolation ends for each missing value
    interp_loc_end = np.searchsorted(valid_loc, empty_loc)
    interp_loc_start = interp_loc_end - 1

    # The indices (e.g. time) of the interpolation endpoints
    interp_t_start = valid_index[interp_loc_start]
    interp_t_end = valid_index[interp_loc_end]

    # The share of the distance between the two endpoints represented by each index location
    share_of_distance = (empty_index - interp_t_start)/(interp_t_end - interp_t_start)

    # Now apply to values, 1 column at a time
    newcolors = []
    valsT = vals.transpose()
    for column in valsT:

        # Select the valid values
        valid_vals = column[valid_loc]

        # These are the actual values of the interpolation ends
        interp_q_start = valid_vals[interp_loc_start]
        interp_q_end = valid_vals[interp_loc_end]

        newcolors.append(colorFader_array(interp_q_start, interp_q_end, mix=share_of_distance))

    newcolors = np.array(newcolors)
    newvals = newcolors.transpose()

    # Put the interpolated values into place
    interpolated_df = df.copy()
    interpolated_df.iloc[empty_loc] = newvals

    return interpolated_df
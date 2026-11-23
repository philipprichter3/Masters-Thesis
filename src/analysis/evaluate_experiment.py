
import math
import warnings
from html import escape

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib import rcParams
from matplotlib.colors import to_hex

#simple interactable plotter class for experiments, written mainly by LLM

class ExperimentPlotter:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(
        self,
        experiments,
        experiment_histories,
        max_cols=3,
        fig_width=5,
        fig_height=3.5,
    ):
        self.experiments = experiments
        self.histories = experiment_histories

        self.max_cols = max_cols
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.keys = list(dict.fromkeys(
            key
            for histories in self.histories.values()
            for results in histories
            for entry in results
            for key in entry
            if key != "time"
        ))

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(self.experiments))
        )

        self.colors = {
            experiment_id: to_hex(color)
            for experiment_id, color in zip(self.experiments, colors)
        }

        self.legend_fontsize = rcParams["legend.fontsize"]
        if isinstance(self.legend_fontsize, str):
            self.legend_fontsize = 10

        with plt.ioff():
            self.fig = plt.figure()

        self.artists = {}
        self.axes_by_key = {}
        self.average_artists = {}

        self.plot_data = self._prepare_data()

        self.key_checkboxes = {}
        self.highlight_key = None
        self.highlight_dropdown = widgets.Dropdown(
            options=[("None", None)] + [(str(key), key) for key in self.keys],
            value=None,
            description="Highlight max of:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", min_width="260px"),
        )
        self.highlight_dropdown.observe(self._on_highlight_change, names="value")
        self.winner_label = widgets.HTML(value="")
        self.average_checkbox = widgets.Checkbox(
            value=False, description="Display average ± std", indent=False,
        )
        self.average_checkbox.observe(self._on_average_change, names="value")
        self.experiment_checkboxes = {}

        self.key_controls = self._create_key_controls()
        self.experiment_controls = self._create_experiment_controls()
        self.save_controls = self._create_save_controls()

        self.update_plot()

# ======================================================================
# ============================= DATA ============================
# ======================================================================

    def _prepare_data(self):

        plot_data = {}

        for experiment_id in self.experiments:

            if experiment_id not in self.histories:
                continue

            histories = self.histories[experiment_id]

            if not histories:
                continue

            # Common time grid across all trajectories
            times = np.unique(np.concatenate([
                np.array([entry["time"] for entry in results],
                        dtype=float)
                for results in histories
                if results
            ]))

            plot_data[experiment_id] = {}

            for key in self.keys:

                aligned_values = []

                for results in histories:

                    if not results:
                        continue

                    t = np.array([
                        entry["time"]
                        for entry in results
                    ], dtype=float)

                    y = np.array([
                        entry.get(key, np.nan)
                        for entry in results
                    ], dtype=float)

                    # Ignore missing or non-finite observations
                    valid = np.isfinite(t) & np.isfinite(y)

                    aligned = np.full(
                        len(times), np.nan, dtype=float
                    )

                    if np.count_nonzero(valid) >= 2:

                        aligned = np.interp(
                            times,
                            t[valid],
                            y[valid],
                            left=np.nan,
                            right=np.nan
                        )

                    elif np.count_nonzero(valid) == 1:

                        idx = np.flatnonzero(
                            np.isclose(times, t[valid][0])
                        )

                        aligned[idx] = y[valid][0]

                    aligned_values.append(aligned)

                if not aligned_values:
                    continue

                values = np.vstack(aligned_values)

                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", RuntimeWarning
                    )

                    mean = np.nanmean(values, axis=0)
                    std = np.nanstd(values, axis=0)

                plot_data[experiment_id][key] = (
                    times, mean, std
                )

        return plot_data

# ======================================================================
# ============================= PLOTTING ============================
# ======================================================================

    def _visible_experiments(self):
        return [
            eid for eid, cfg in self.experiments.items()
            if cfg["plot"] and eid in self.plot_data
        ]

    def _highlight_winner(self):
        """Find the highest last mean in the selected quantity."""
        key = self.highlight_key
        if key is None or not self.key_checkboxes[key].value:
            return None

        candidates = []
        for eid in self._visible_experiments():
            data = self.plot_data[eid].get(key)
            if data is None:
                continue
            mean = data[1]
            if len(mean) and np.isfinite(mean[-1]):
                candidates.append((float(mean[-1]), eid))
        return max(candidates, key=lambda pair: pair[0])[1] if candidates else None

    def _aggregate_visible(self, key, experiment_ids):
        """Mean and population std of experiment means, aligned in time.

        The first visible experiment supplies the time grid. Other curves
        are linearly interpolated onto it, with no extrapolation.
        """
        curves = []
        reference_times = None
        for eid in experiment_ids:
            data = self.plot_data[eid].get(key)
            if data is None:
                continue
            times, means, _ = data
            times = np.asarray(times, dtype=float)
            means = np.asarray(means, dtype=float)
            valid = np.isfinite(times) & np.isfinite(means)
            if not np.any(valid):
                continue
            times, means = times[valid], means[valid]
            order = np.argsort(times)
            times, means = times[order], means[order]
            times, unique_indices = np.unique(times, return_index=True)
            means = means[unique_indices]
            if reference_times is None:
                reference_times = times
            if len(times) == 1:
                aligned = np.full(reference_times.shape, np.nan)
                aligned[np.isclose(reference_times, times[0])] = means[0]
            else:
                aligned = np.interp(
                    reference_times, times, means, left=np.nan, right=np.nan
                )
            curves.append(aligned)

        if not curves:
            return None
        values = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        return reference_times, avg, std

    def update_plot(self):
        visible_keys = [
            key for key in self.keys if self.key_checkboxes[key].value
        ]
        visible_experiments = self._visible_experiments()
        winner = self._highlight_winner()

        # Keep the winner identification in sync with selection and visibility.
        winner_text = None
        if self.highlight_key is not None:
            if winner is not None:
                final_value = float(self.plot_data[winner][self.highlight_key][1][-1])
                winner_text = (
                    f"Maximum for {self.highlight_key}: "
                    f"{self.experiments[winner]['name']} "
                    f"(final mean = {final_value:.6g})"
                )
                self.winner_label.value = (
                    '<span style="color:#c62828;font-weight:bold">'
                    + escape(winner_text) + '</span>'
                )
            else:
                self.winner_label.value = (
                    '<span style="color:#777">No visible experiment has '
                    'a finite final value for the selected quantity.</span>'
                )
        else:
            self.winner_label.value = ""

        self.fig.clear()
        self.artists.clear()
        self.axes_by_key.clear()
        self.average_artists.clear()

        if not visible_keys:
            self.fig.canvas.draw_idle()
            return

        ncols = min(self.max_cols, len(visible_keys))
        nrows = math.ceil(len(visible_keys) / ncols)
        self.fig.set_size_inches(
            self.fig_width * ncols, self.fig_height * nrows
        )
        axes = self.fig.subplots(nrows, ncols, sharex=True, squeeze=False)

        for ax, key in zip(axes.flat, visible_keys):
            self.axes_by_key[key] = ax

            for experiment_id, config in self.experiments.items():
                data = self.plot_data.get(experiment_id, {}).get(key)
                if data is None:
                    continue
                times, mean, std = data
                visible = config["plot"]
                average_mode = self.average_checkbox.value
                color = (
                    "#aaaaaa" if average_mode else
                    self.colors[experiment_id] if winner is None else
                    "red" if experiment_id == winner else "#aaaaaa"
                )
                alpha = (
                    0.12 if average_mode else
                    0.25 if winner is None or experiment_id == winner else 0.12
                )
                line, = ax.plot(
                    times, mean, label=config["name"], ls="--",
                    color=color, linewidth=1.5,
                    zorder=4 if experiment_id == winner else 2,
                    visible=visible,
                )
                band = ax.fill_between(
                    times, mean - std, mean + std,
                    alpha=alpha, color=color, visible=visible,
                    zorder=3 if experiment_id == winner else 1,
                )
                self.artists.setdefault(experiment_id, []).append((line, band))

            if self.average_checkbox.value:
                aggregate = self._aggregate_visible(key, visible_experiments)
                if aggregate is not None:
                    times, mean, std = aggregate
                    avg_line, = ax.plot(
                        times, mean, color="red", ls="-", lw=1.5,
                        label="Average ± std", zorder=6,
                    )
                    avg_band = ax.fill_between(
                        times, mean - std, mean + std,
                        color="red", alpha=0.20, zorder=5,
                    )
                    self.average_artists[key] = (avg_line, avg_band)

            ax.set_title(key)
            ax.set_xlabel("time")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(visible_keys):]:
            ax.set_visible(False)
        # Figure annotation is included in image exports, unlike widgets.
        if winner_text is not None:
            self.fig.suptitle(winner_text, color="red", fontsize=11, fontweight="bold")
            self.fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            self.fig.tight_layout()
        self.fig.canvas.draw_idle()

# ======================================================================
# ============================= QUNATITY CONTROLS ============================
# ======================================================================

    def _create_key_controls(self):
        for key in self.keys:
            checkbox = widgets.Checkbox(
                value=False, description=key, indent=False,
                layout=widgets.Layout(width="auto"),
            )
            checkbox.observe(self._on_key_change, names="value")
            self.key_checkboxes[key] = checkbox

        return widgets.GridBox(
            list(self.key_checkboxes.values()),
            layout=widgets.Layout(
                grid_template_columns="repeat(3, max-content)",
                justify_content="center", grid_gap="4px 20px",
                padding="8px 12px", border="1px solid #cccccc",
                width="auto",
            ),
        )

    def _on_key_change(self, change):
        # Hiding the selected criterion resets the maximum highlighting.
        if (self.highlight_key is not None
                and not self.key_checkboxes[self.highlight_key].value):
            self.highlight_dropdown.value = None
        self.update_plot()

    def _on_average_change(self, change):
        if change["new"] and self.highlight_key is not None:
            self.highlight_dropdown.value = None
        self.update_plot()

    def _on_highlight_change(self, change):
        key = change["new"]
        self.highlight_key = key
        if key is not None:
            # The average and maximum highlighting modes are exclusive.
            if self.average_checkbox.value:
                self.average_checkbox.value = False
            # Keep the criterion subplot visible automatically.
            if not self.key_checkboxes[key].value:
                self.key_checkboxes[key].value = True
        self.update_plot()

# ======================================================================
# ============================= EXPERIMENT CONTROLS ============================
# ======================================================================

    def _create_experiment_controls(self):

        rows = []

        for experiment_id, config in self.experiments.items():

            if experiment_id not in self.plot_data:
                continue

            color = self.colors[experiment_id]

            color_indicator = widgets.HTML(
                value=f"""
                <div style="
                    width: 26px;
                    height: 0;
                    border-top: 2px dashed {color};
                    margin: 0 6px 0 2px;
                "></div>
                """,
                layout=widgets.Layout(width="36px"),
            )

            checkbox = widgets.Checkbox(
                value=config["plot"],
                indent=False,
                layout=widgets.Layout(
                    width="25px",
                    margin="0",
                ),
            )

            self.experiment_checkboxes[experiment_id] = checkbox

            checkbox.observe(
                lambda change, eid=experiment_id:
                    self._toggle_experiment(eid, change["new"]),
                names="value",
            )

            label = widgets.HTML(
                value=f"""
                <span style="
                    font-family: Arial, sans-serif;
                    font-size: {self.legend_fontsize}pt;
                    white-space: nowrap;
                ">
                    {config["name"]}
                </span>
                """
            )

            row = widgets.HBox(
                [color_indicator, checkbox, label],
                layout=widgets.Layout(
                    align_items="center",
                    width="auto",
                    gap="2px",
                    padding="2px 6px",
                ),
            )

            rows.append(row)

        return widgets.GridBox(
            rows,
            layout=widgets.Layout(
                grid_template_columns="repeat(2, max-content)",
                justify_content="center",
                grid_gap="4px 20px",
                padding="8px 12px",
                border="1px solid #cccccc",
                width="auto",
            ),
        )

    def _toggle_experiment(self, experiment_id, visible):
        self.experiments[experiment_id]["plot"] = visible
        # Recompute the aggregate and the maximum whenever visibility changes.
        self.update_plot()

# ======================================================================
# ============================= SAVE ============================
# ======================================================================

    def save(self, filename="experiments.png"):
        """Save the plots with a separate legend footer, without compressing panels.

        The winner is identified by '(MAX)' in the legend, but its explanatory
        text heading is not exported. The notebook figure is restored afterward.
        """
        if not self.axes_by_key:
            print("No quantities selected.")
            return

        winner = self._highlight_winner()
        handles, labels = [], []
        for experiment_id, config in self.experiments.items():
            if experiment_id not in self.artists or not config["plot"]:
                continue
            line, _ = self.artists[experiment_id][0]
            label = str(config["name"])
            if experiment_id == winner:
                label += " (MAX)"
            handles.append(line)
            labels.append(label)

        if self.average_checkbox.value and self.average_artists:
            first_key = next(iter(self.average_artists))
            avg_line, _ = self.average_artists[first_key]
            handles.append(avg_line)
            labels.append("Average ± std")

        old_positions = {ax: ax.get_position().frozen() for ax in self.fig.axes}
        old_size = self.fig.get_size_inches().copy()
        suptitle = self.fig._suptitle
        suptitle_visible = suptitle.get_visible() if suptitle else False
        legend = None

        try:
            # The winner remains named in the notebook UI, but not as a heading
            # in the exported PNG. Only the legend includes the '(MAX)' suffix.
            if suptitle is not None:
                suptitle.set_visible(False)

            if handles:
                legend = self.fig.legend(
                    handles, labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=min(2, len(handles)),
                    frameon=True,
                    fontsize=self.legend_fontsize,
                    handlelength=2.5,
                    columnspacing=2.0,
                )
                self.fig.canvas.draw()
                renderer = self.fig.canvas.get_renderer()
                legend_height_inches = (
                    legend.get_window_extent(renderer=renderer).height
                    / self.fig.dpi
                )
            else:
                legend_height_inches = 0.0

            # Add *physical height* for the legend instead of taking it from
            # the plot area. This preserves the subplots' vertical dimensions.
            footer_inches = legend_height_inches + 0.45 if legend else 0.0
            new_height = old_size[1] + footer_inches
            self.fig.set_size_inches(old_size[0], new_height, forward=False)
            bottom = footer_inches / new_height if footer_inches else 0.0
            self.fig.tight_layout(rect=(0, bottom, 1, 0.99))
            self.fig.savefig(
                filename, dpi=300, bbox_inches="tight", pad_inches=0.20,
                bbox_extra_artists=[legend] if legend else None,
            )
        finally:
            if legend is not None:
                legend.remove()
            self.fig.set_size_inches(old_size, forward=False)
            for ax, pos in old_positions.items():
                ax.set_position(pos)
            if suptitle is not None:
                suptitle.set_visible(suptitle_visible)
            self.fig.canvas.draw_idle()

        print(f"Saved: {filename}")

    def _create_save_controls(self):

        button = widgets.Button(
            description="Save",
            icon="save",
        )

        button.on_click(
            lambda _: self.save("experiments.png")
        )

        return widgets.HBox(
            [button],
            layout=widgets.Layout(
                justify_content="center",
                gap="10px",
            ),
        )

# ======================================================================
# ============================= DISPLAY ============================
# ======================================================================

    def show(self):

        display(
            widgets.VBox(
                [
                    self.fig.canvas,

                    widgets.HTML("<b>Displayed quantities</b>"),
                    self.key_controls,
                    self.highlight_dropdown,
                    self.winner_label,

                    self.average_checkbox,

                    widgets.HTML("<b>Experiments</b>"),
                    self.experiment_controls,

                    self.save_controls,
                ],
                layout=widgets.Layout(
                    align_items="center",
                    gap="8px",
                ),
            )
        )

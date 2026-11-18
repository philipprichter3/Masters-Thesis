
import math
import warnings

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

        self.plot_data = self._prepare_data()

        self.key_checkboxes = {}
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

            if not histories or not histories[0]:
                continue

            plot_data[experiment_id] = {}

            times = np.array([
                entry["time"]
                for entry in histories[0]
            ])

            for key in self.keys:

                values = np.array([
                    [
                        entry.get(key, np.nan)
                        for entry in results
                    ]
                    for results in histories
                ], dtype=float)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(values, axis=0)
                    std = np.nanstd(values, axis=0)

                plot_data[experiment_id][key] = (
                    times, mean, std
                )

        return plot_data

# ======================================================================
# ============================= PLOTTING ============================
# ======================================================================

    def update_plot(self):

        visible_keys = [
            key
            for key in self.keys
            if self.key_checkboxes[key].value
        ]

        self.fig.clear()
        self.artists.clear()
        self.axes_by_key.clear()

        if not visible_keys:
            self.fig.canvas.draw_idle()
            return

        ncols = min(self.max_cols, len(visible_keys))
        nrows = math.ceil(len(visible_keys) / ncols)

        self.fig.set_size_inches(
            self.fig_width * ncols,
            self.fig_height * nrows,
        )

        axes = self.fig.subplots(
            nrows,
            ncols,
            sharex=True,
            squeeze=False,
        )

        for ax, key in zip(axes.flat, visible_keys):

            self.axes_by_key[key] = ax

            for experiment_id, config in self.experiments.items():

                if experiment_id not in self.plot_data:
                    continue

                times, mean, std = (
                    self.plot_data[experiment_id][key]
                )

                color = self.colors[experiment_id]
                visible = config["plot"]

                line, = ax.plot(
                    times,
                    mean,
                    label=config["name"],
                    ls="--",
                    color=color,
                    visible=visible,
                )

                band = ax.fill_between(
                    times,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    color=color,
                    visible=visible,
                )

                self.artists.setdefault(
                    experiment_id, []
                ).append((line, band))

            ax.set_title(key)
            ax.set_xlabel("time")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(visible_keys):]:
            ax.set_visible(False)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

# ======================================================================
# ============================= QUNATITY CONTROLS ============================
# ======================================================================

    def _create_key_controls(self):

        for key in self.keys:

            checkbox = widgets.Checkbox(
                value=False,
                description=key,
                indent=False,
                layout=widgets.Layout(width="auto"),
            )

            checkbox.observe(
                self._on_key_change,
                names="value",
            )

            self.key_checkboxes[key] = checkbox

        return widgets.GridBox(
            list(self.key_checkboxes.values()),
            layout=widgets.Layout(
                grid_template_columns="repeat(3, max-content)",
                justify_content="center",
                grid_gap="4px 20px",
                padding="8px 12px",
                border="1px solid #cccccc",
                width="auto",
            ),
        )

    def _on_key_change(self, change):
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

        for line, band in self.artists.get(experiment_id, []):
            line.set_visible(visible)
            band.set_visible(visible)

        self.fig.canvas.draw_idle()

# ======================================================================
# ============================= SAVE ============================
# ======================================================================

    def save(self, filename="experiments.png"):

        if not self.axes_by_key:
            print("No quantities selected.")
            return

        handles = []
        labels = []

        for experiment_id, config in self.experiments.items():

            if experiment_id not in self.artists:
                continue

            if not config["plot"]:
                continue

            line, _ = self.artists[experiment_id][0]

            handles.append(line)
            labels.append(config["name"])

        legend = None

        if handles:
            legend = self.fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
                frameon=True,
                fontsize=self.legend_fontsize,
                handlelength=2.5,
                columnspacing=2.0,
            )

        try:
            self.fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                bbox_extra_artists=[legend] if legend else None,
            )
        finally:
            if legend is not None:
                legend.remove()

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

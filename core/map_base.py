from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button

from abc import ABC, abstractmethod


from core.progress import ProgressBar


__all__ = ["MapBase"]


class MapBase(ABC):
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        if not isinstance(value, dict):
            raise TypeError("Params can only be a dict!")

        if len(value) > 2:
            raise ValueError("Having more than 2 parameters is not supported!")

        if len(value) == 1:
            self._plotter = self._plot2D
        elif len(value) == 2:
            self._plotter = self._plot3D

        self._params = value

    @property
    def vars(self):
        return self._vars

    @vars.setter
    def vars(self, value):
        if not isinstance(value, dict):
            raise TypeError("Vars can only be a dict!")

        self._vars = value

    def __init__(self, last_num=100):
        self.name = "Map"

        self.last_num = last_num
        self.last_iters = []

        self.params = {}
        self.vars = {}

        self._figure = None
        self.anim = None
        self.cid = None

    @abstractmethod
    def iterate(self, *args):
        pass

    @ProgressBar(width=32, step=0.032347)
    def iterate_field(self, iterations):
        self.last_iters = []

        print(f"Iterating {self.name}...")

        x = list(self.vars.values())

        for i in range(iterations):
            x = self.iterate(*x)

            if i >= (iterations - self.last_num):
                self.last_iters.append(x)

            yield i / iterations

    def plot_bifurcation_diagram(self, iterations):
        self._figure = plt.figure()
        self._figure.suptitle(f"{self.name} Bifurcation Diagram")

        self.iterate_field(iterations)
        self._plotter()

    def _plot2D(self):
        param_keys = list(self.params.keys())
        param_vals = list(self.params.values())

        x = param_vals[0]

        for ind, v_name in enumerate(self.vars):
            ax = self._figure.add_subplot(1, len(self.vars), ind + 1)

            ax.set_xlabel(param_keys[0])
            ax.set_ylabel(v_name)

            ax.set_xlim(x[0], x[-1])

            ax.grid(True)

            for y in self.last_iters:
                ax.plot(x.flatten(), y[ind].flatten(), ',k', alpha=0.25)

    def _plot3D(self):
        var_sct = []

        param_keys = list(self.params.keys())
        param_vals = list(self.params.values())

        x = param_vals[0]
        y = param_vals[1]

        middle_x = int(x[0].size / 2)

        axparam = plt.axes([0.08, 0.15, 0.03, 0.6])
        sparam = Slider(axparam,
                        param_keys[0],
                        x[0][0],
                        x[0][-1],
                        valinit=x[0][middle_x],
                        valstep=5 * (x[0][1] - x[0][0]),
                        orientation="vertical")

        def update(*args):
            p = sparam.val
            ind = int(x[0].size * (p - x[0][0]) / (x[0][-1] - x[0][0]))

            for v_ind, sct_a in enumerate(var_sct):
                for i, sct in enumerate(sct_a):
                    sct.set_data(x[:, ind].flatten(), y[:, ind].flatten())
                    sct.set_3d_properties(self.last_iters[i][v_ind][:, ind].flatten())

        self.cid = sparam.on_changed(update)
        sparam.drag_active = False

        def anim_plot(frame):
            ind = int(x[0].size * frame / 100)

            for v_ind, sct_a in enumerate(var_sct):
                for i, sct in enumerate(sct_a):
                    sct.set_data(x[:, ind].flatten(), y[:, ind].flatten())
                    sct.set_3d_properties(self.last_iters[i][v_ind][:, ind].flatten())

            return var_sct.flatten()

        def clear_plot(*args):
            ind = 0

            for v_ind, sct_a in enumerate(var_sct):
                for i, sct in enumerate(sct_a):
                    sct.set_data(x[:, ind].flatten(), y[:, ind].flatten())
                    sct.set_3d_properties(self.last_iters[i][v_ind][:, ind].flatten())

            return var_sct.flatten()

        def start_anim(*args):
            sparam.disconnect(self.cid)

            if self.anim is None:
                self.anim = animation.FuncAnimation(self._figure, anim_plot, 100, init_func=partial(clear_plot, 0), interval=20)
            else:
                self.anim.event_source.start()

        def stop_anim(*args):
            self.cid = sparam.on_changed(update)

            if self.anim is not None:
                self.anim.event_source.stop()

        axstart = plt.axes([0.75, 0.05, 0.1, 0.075])
        axstop = plt.axes([0.86, 0.05, 0.1, 0.075])
        self.bstart = Button(axstart, 'Start')
        self.bstart.on_clicked(start_anim)
        self.bstop = Button(axstop, 'Stop')
        self.bstop.on_clicked(stop_anim)

        for ind, v_name in enumerate(self.vars):
            ax = self._figure.add_subplot(1, len(self.vars), ind + 1, projection='3d')

            ax.set_xlabel(param_keys[0])
            ax.set_ylabel(param_keys[1])
            ax.set_zlabel(v_name)

            ax.set_xlim3d(x[0][0], x[0][-1])
            ax.set_ylim3d(y[0][0], y[-1][0])

            inner_vals = [it[ind] for it in self.last_iters]
            ax.set_zlim3d(np.min(inner_vals), np.max(inner_vals))

            ax.grid(True)

            var_sct.append([])

            for z in self.last_iters:
                sct, = ax.plot3D(x[:,middle_x].flatten(), y[:,middle_x].flatten(), z[ind][:,middle_x].flatten(), ',k', alpha=0.25)
                var_sct[ind].append(sct)

        var_sct = np.array(var_sct)


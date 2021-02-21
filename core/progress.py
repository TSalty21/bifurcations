import functools

__all__ = ["ProgressBar"]


class ProgressBarPrinter:
    def __init__(self, decorated, width, step):
        self._decorated = decorated

        self.width = width
        self.step = step

        self.current_progress = 0
        self.last_progress = 0

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        progress_generator = self._decorated(*args, **kwargs)
        try:
            while True:
                progress = next(progress_generator)
                self.update(progress)
        except StopIteration as result:
            self.current_progress = 0
            self.last_progress = 0

            self.print_progress(1)
            print('\n')

            return result.value

    def update(self, progress):
        self.current_progress = progress

        if self.current_progress - self.last_progress >= self.step:
            self.last_progress = self.current_progress
            self.print_progress(self.current_progress)

    def print_progress(self, progress):
        percentage = int(100 * progress)
        bar_width = int(self.width * progress)
        print(f"\rProgress: [{('=' * bar_width).ljust(self.width)}] {percentage}%", end='')


def ProgressBar(width=30, step=0.05):
    def printer(function):
        return ProgressBarPrinter(function, width, step)

    return printer
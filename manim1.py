from manim import *
import first

class TestGraph(GraphScene):

    arguments = {
        "graph_origin": [5, 5, 0],#ORIGIN,
        "function_color": WHITE,
        "axes_color": BLUE
    }

    def __init__(self, **kwargs):
        print('KWARGS')
        print(kwargs)
        kwargs.update({
            'x_min': 0,
            'x_max': 4,
            'y_min': -3,
            'y_max': 25,
            'graph_origin': [0, 0, 0]
        })
        super().__init__(**kwargs)

    def construct(self):
        #self.x_min = -2
        #self.x_max = 2

        #self.x_axis_config.update({
        #    'x_min': self.x_min,
        #    'x_max': self.x_max
        #})
        '''
        self.y_axis_config.update({
            #'x_min': -5,
            #'x_max': 5
        })
        '''

        def fun(x):
            return first.lerp(x)
            if .5 < x < .9:
                return float('nan')
            return 0.8*x**3

        def fun2(x):
            return 0.2*x**4 - 2


        self.setup_axes(animate=True)
        function_color = WHITE

        print('HI')
        print(ORIGIN)
        print(self.x_min)
        print(self.x_max)

        func_graph = self.get_graph(fun, function_color)#, x_min=self.x_min, x_max=self.x_max)
        graph_lab = self.get_graph_label(func_graph, label="x^{2}")

        func_graph2 = self.get_graph(fun2, function_color)

        self.wait(2)
        self.play(ShowCreation(func_graph), Write(graph_lab))
        self.wait(2)
        self.play(Transform(func_graph, func_graph2))
        self.wait(2)

class Plot1(GraphScene):
    def construct(self):
        self.setup_axes()
        func_graph = self.get_graph(lambda x: np.sin(x))
        #self.add(func_graph)
        self.play(ShowCreation(func_graph))

        self.wait(2)
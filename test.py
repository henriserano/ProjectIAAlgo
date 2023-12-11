class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"MyClass instance with x={self.x} and y={self.y}"

# Create an instance of MyClass
obj = [MyClass(10, 20)]

# When using print or str function, __str__ is called
print(obj[0])
package com.moofiyTv.mossapp.OCP_DI.correct;

public class Calculator {
    double result;

    void calculate(Operation operation) {
        result = operation.operate();
    }

    public static void main(String[] args) {
        Calculator calculator = new Calculator();
        Addition addition = new Addition(5, 6);
        calculator.calculate(addition);
        Substraction substraction = new Substraction(4, 3);
        calculator.calculate(substraction);
    }

}

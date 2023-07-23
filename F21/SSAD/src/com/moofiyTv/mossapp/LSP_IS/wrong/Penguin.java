package com.moofiyTv.mossapp.LSP_IS.wrong;

import com.moofiyTv.mossapp.LSP_IS.correct.Flyable;

public class Penguin implements Bird {
    @Override
    public void fly() {
        throw new AssertionError("I Can't fly");
    }

    @Override
    public void eat() {
        System.out.println("eating");
    }

    @Override
    public void layEggs() {
        System.out.println("laying eggs");
    }

    @Override
    public void swim() {
        System.out.println("swim");

    }
}

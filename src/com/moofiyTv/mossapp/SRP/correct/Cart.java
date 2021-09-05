package com.moofiyTv.mossapp.SRP.correct;

import com.moofiyTv.mossapp.SRP.Product;

import java.util.List;


/**
 * ✅
 * Follows SRP : only presents a simple object!
 */
public class Cart {

    List<Product> products;
    double totals;
    String token;

    public List<Product> getProducts() {
        return products;
    }

    public double getTotals() {
        return totals;
    }
}


package com.moofiyTv.mossapp.SRP.correct;

import com.moofiyTv.mossapp.SRP.Product;

import java.util.List;

public class CartOperations {

    List<Product> products;

    void addToCart(Product product) {
        products.add(product);
    }

    void removeFromCart(Product product) {
        products.remove(product);
    }

}

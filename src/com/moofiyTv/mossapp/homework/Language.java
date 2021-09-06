package com.moofiyTv.mossapp.homework;

public class Language implements ILanguage{
    private String language;

    public Language(String language){
        this.language = language;
    }

    @Override
    public String getName() {
        return language;
    }

    @Override
    public String translateTo(String destinationLanguage, String phrase) {
        return null;
    }
    
}

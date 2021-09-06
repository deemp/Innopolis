package com.moofiyTv.mossapp.homework;


import java.util.List;

// lab work
// This Class violates all SOLID Principles, fix it in a Logical way ;


public class Human implements 
    INeeds,
    IKnowsLanguage,
    IWithHobby, 
    IWithNickname,
    IWithSalary
{
    private List<IHobby> hobbies;
    
    @Override
    public int addHobby(IHobby hobby) {
        hobbies.add(hobby);
        return hobbies.size();
    }

    @Override
    public List<IHobby> getHobbies() {
        return hobbies;
    }


    private double salary;
    @Override
    public void calculateTax(int percentage) {
        salary = salary * percentage;
    }

    @Override
    public String sayHello(String language) {
        if (language == "Arabic")
            return "مرحبا";
        else
            return "Hello";
    }

    private String name;
    private String nickName;
    
    @Override
    public void setNickname(String postFix) {
        this.nickName = name.concat(postFix);
    }
    
    @Override
    public String getNickname() {
        return nickName;        
    }

    @Override
    public void pray() {
        // TODO Auto-generated method stub
    }

    @Override
    public void playSports() {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void getMarried() {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void ownCompany() {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void becomeEmployee() {
        // TODO Auto-generated method stub
        
    }
}
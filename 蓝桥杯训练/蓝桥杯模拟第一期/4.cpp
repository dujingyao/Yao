#include<iostream>
using namespace std;

int two(int year){
    if(year%400==0||year%100!=0&&year%4==0) return 29;
    return 28;
}

int month1(int year,int x){
    switch(x){
        case 1:return 31;
        case 2:return two(year);
        case 3:return 31;
        case 4:return 30;
        case 5:return 31;
        case 6:return 30;
        case 7:return 31;
        case 8:return 31;
        case 9:return 30;
        case 10:return 31;
        case 11:return 30;
        case 12:return 31;
    }
}

int main(){
    int year=1901,month=1,day=1,week=2;
    int res=0;
    while(year<2025){
        if(day%10==1&&week==1) res++;
        day++;
        week++;
        if(week==8) week=1;
        if(day>month1(year,month)){
            day=1;
            month++;
        }
        if(month==13){
                month=1;
                year++;
        }
    }
    cout<<res<<endl;
    return 0;
}
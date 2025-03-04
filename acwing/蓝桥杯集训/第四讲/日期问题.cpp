#include<bits/stdc++.h>
using namespace std;

//日期范围是1960.1.1-2059.12.31

bool check(int year,int month,int day){
    if(month==0||month>=13) return false;
    if(month==2){
        if(year%4==0&&year%100!=0||year%400==0){//闰年29
            if(day>=30||day==0) return false;
            else return true;
        }
        else{
            if(day>=29||day==0) return false;
            else return true;
        }
    }
    if(month==1||month==3||month==5||month==7||month==8||month==10||month==12){
        if(day>=32||day==0) return false;
        else return true;
    }
    else{
        if(day>=31||day==0) return false;
        else return true;
    }
}

int main(){
    
    int a, b, c;
    scanf("%d/%d/%d", &a, &b, &c);
    for(int i=19600101;i<=20591231;i++){
        int year=i/10000,month=i/100%100,day=i%100;
        if(check(year,month,day)){
            if (year % 100 == a && month == b && day == c ||        // 年/月/日
                month == a && day == b && year % 100 == c ||        // 月/日/年
                day == a && month == b &&year % 100 == c)           // 日/月/年
                printf("%d-%02d-%02d\n", year, month, day);
        }
    }
    
    
    return 0;
}
#include<iostream>
#include<vector>
using namespace std;

int main(){
    vector<char> a;
    while(1){
        char ch;
        scanf("%c",&ch);
        if(ch=='\n') break;
        a.push_back(ch);
    }
    int x=a[a.size()-1]-'0';
    if(x%2==0) printf("even");
    else printf("odd");
    
    return 0;
}
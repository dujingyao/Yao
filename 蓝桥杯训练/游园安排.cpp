#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;

bool cmp(const string & a,const string & b){
    return a+b<b+a;
}

int main(){
    string input;
    cin>>input;
    vector<string> name;
    int start=0;

    for(int i=0;i<input.size();i++){
        if(i==input.size()-1||isupper(input[i+1])){
            name.push_back(input.substr(start,i-start+1));
            start=i+1;
        }
    }

    vector<int> dp(name.size(),1);
    vector<int> prev(name.size(),-1);

    int maxLen=1, maxIdx=0;
    for(int i=1;i<name.size();i++){
        for(int j=0;j<i;j++){
            if(cmp(name[j],name[i])&&dp[j]+1>dp[i]){   //&& dp[j] + 1 > dp[i]
                dp[i]=dp[j]+1;
                prev[i]=j;
            }
        }
        if(dp[i]>maxLen){
            maxLen=dp[i];
            maxIdx=i;
        }
    }

    vector<string> lis;
    while(maxIdx!=-1){
        lis.push_back(name[maxIdx]);
        maxIdx=prev[maxIdx];
    }

    vector<bool> used(name.size(),false);
    reverse(lis.begin(),lis.end());
    for(const auto &names:lis){
        for(int i=0;i<name.size();i++){
            if(name[i]==names&&!used[i]){
                cout<<names;
                used[i]=true;
                break;
            }
        }
    }



    return 0;
}
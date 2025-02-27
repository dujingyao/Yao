#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

int main(){
    int n;
    cin>>n;
    while(n--){
        int m;
        cin>>m;
        vector<int> b;
        for(int i=1;i<=m/i;i++){
            if(m%i==0){
                b.push_back(i);
                if(m/i!=i){
                    b.push_back(m/i);
                }
            }
        }
        sort(b.begin(),b.end());
        for(int i=0;i<b.size();i++){
            cout<<b[i]<<' ';
        }
        cout<<endl;
    }
    return 0;
}
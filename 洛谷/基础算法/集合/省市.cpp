#include<bits/stdc++.h>
using namespace std;

typedef pair<int,int> PII;

const int N=2e5+10,mod=233333;

char a[N],b[N];
long long ans;
vector<PII> linker[mod+2];

int gethash(char a[],char b[]){
    return a[0]-'A'+(a[1]-'A')*26+(b[0]-'A')*26*26+(b[1]-'A')*26*26*26;
}

void insert(int x){
    for(int i=0;i<linker[x%mod].size();i++){
        if(linker[x%mod][i].first==x){
            linker[x%mod][i].second++;
            break;
        }
    }
    linker[x%mod].push_back(PII(x,1));
}
int find(int x){
    for(int i=0;i<linker[x%mod].size();i++){
        if(linker[x%mod][i].first==x){
            return linker[x%mod][i].second;
        }
    }
    return 0;
}

int main(){
    int n;
    cin>>n;
    while(n--){
        cin>>a>>b;
        a[2]=0;
        if(a[0]!=b[0]||a[1]!=b[1]){
            ans+=find(gethash(b,a));
        }
        insert(gethash(a,b));
    }
    cout<<ans<<endl;
    return 0;
}
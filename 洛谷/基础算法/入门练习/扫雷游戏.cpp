#include<iostream>
using namespace std;
int n,m;
char check(char a[][105],int col,int row){
    int sum=0;
    for(int i=col-1;i<=col+1;i++){
        for(int j=row-1;j<=row+1;j++){
            if(a[col][row]=='*') return '*';
            if(i==col&&j==row) continue;
            if(a[i][j]=='*') sum++;
        }
    }
    return sum+'0';
}

int main(){
    cin>>n>>m;
    char a[105][105];

    for(int i=0;i<=m;i++){
        a[0][i]='?';
        a[n+1][i]='?';
    }
    for(int i=0;i<=n;i++){
        a[i][0]=='?';
        a[i][m+1]=='?';
    }


    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            cin>>a[i][j];
        }
    }

    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            cout<<check(a,i,j);
        }
        cout<<endl;
    }

    return 0;
}
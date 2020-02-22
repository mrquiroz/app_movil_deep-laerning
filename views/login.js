import React, { Component } from 'react';
import { StyleSheet, Text, View, AppRegistry, Linking,Image, ImageBackground, TextInput,Alert} from 'react-native';
import { ThemeProvider,Button } from 'react-native-elements';
import ScalableText from 'react-native-text';

//import { createStackNavigator, createAppContainer } from 'react-navigation';

const theme = {
  Button: {
    titleStyle: {
      color: 'white',
    },
  },
};


export default class Login extends Component {

    constructor(props) {
      super(props);
      this.state = {
        password: '',
        correo:'',
        navegar:props.navigation.navigate,
      }
    }
    hacerLogIn = () => {
        this.props.navigation.navigate('clasificador')
      
  };
    render(){
      let background = require('../assets/login.png');
      let logo = require('../assets/logo.png');
      return (
        <ImageBackground
        source={background}
        style={styles.background}
        blurRadius={1}
        >  
          <View style={styles.container}>
            <View style={styles.logoContainer}>
              <Image source={logo} style={styles.logo}/>
            </View>
            <ScalableText style={styles.text}>Nevos es un proyecto que busca apoyar con una clasificación temprana posibles lesiones a la piel, en caso de que encontremos alguna anomalía te recomendaremos ir a un profesional.</ScalableText>
            <View style = {styles.signIn}>
              <Button
              rounded
              onPress={this.hacerLogIn}
              title="Clasificar"
              color="#245465"
              buttonStyle={styles.myButton}
              style={{borderRadius: 50}}
              />
  
            </View>
            <Text style={styles.text_web} onPress={ ()=> Linking.openURL('https://nevos.cl') } >Sitio Web</Text>
          </View>
         </ImageBackground>
      );
    }
  }
  
  const styles = StyleSheet.create({
    container: {
      flex:1,
    },
    logoContainer: {
      alignItems: 'center',
      marginTop:130,
      marginBottom: 30,
      opacity:0.7,
    },
    loginContainer: {
      justifyContent: 'center',
      flexDirection:'row',
      marginLeft: 30,
      marginRight: 30,
      marginBottom:15,
      borderBottomColor: '#fff',
      borderBottomWidth: 0.5,
      height: 30,
      alignItems: "stretch",
    },
    background: {
      flex:1,
      width: '100%', 
      height: '101%',
      resizeMode: 'cover',
    },
    myButton2:{ 
      borderRadius:400,
      borderColor: '#d6d7da',
    },
    
    logo: {  
      width: 110,
      height: 150,
    },
    input: {
      flex:1,
      color:'#fff',
      paddingRight: 15,
      paddingBottom: 5,
      paddingLeft: 0  
    },
    signIn: {
      backgroundColor: '#46AFA3',
      marginTop: 60,
      marginBottom: 20,
      marginLeft: 20,
      marginRight: 20,
      color: 'transparent',
      borderRadius: 140,
      fontSize: 10,
    },
    signUp: {
      borderColor: 'transparent',
      borderWidth: 0.5,
      marginLeft:20,
      marginRight:20,
      color: '#fff',
      borderRadius: 7
    },
    inputIcon: {
      paddingRight:20,
      paddingLeft:10
    },
    text: {
      fontSize: 16,
      textAlign: 'center',
      color: '#333333',
      marginBottom: 5,
      marginLeft:20,
      marginRight:20,
    },
    text_web: {
      fontSize: 16,
      textAlign: 'center',
      color: 'white',
      marginBottom: 1,
      marginLeft:20,
      marginRight:20,
    },
    myButton:{ 
      backgroundColor:'#68A7B8',
      borderRadius:400,
      borderColor: '#d6d7da',
      
    },
    forgotPassword: {
      color: '#fff',
      marginLeft:178,
      opacity: 0.8,
    }
  });
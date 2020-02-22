import React, { Component } from 'react';
import { createAppContainer} from 'react-navigation';
import { createStackNavigator } from 'react-navigation-stack';
import Login from './views/login.js'
import Clasificador from './clasificador.js'





// para que la logica funcione, cuando se loggen en vez de ir a menu, dirigance a index
const RootStack = createStackNavigator(
  {
    Login :{
      screen: Login,
      navigationOptions: {
        headerShown: false,
      }
    },
    clasificador:{
      screen: Clasificador,
      navigationOptions: {
        headerShown: false,
      }
    }
  },
  {
    initialRouteName: 'Login',
  }
);

const AppContainer = createAppContainer(RootStack);
export default class App extends Component {
  render() {
    return <AppContainer />;
  }
}


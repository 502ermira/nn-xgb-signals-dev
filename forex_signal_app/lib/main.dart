import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  final List<String> timeframes = [
    '7min',
    '15min',
    '1h',
    '4h',
    '1d',
  ];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Forex Signal App',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: SignalSelectorPage(timeframes: timeframes),
      );
  }
}

class SignalSelectorPage extends StatefulWidget {
  final List<String> timeframes;

  const SignalSelectorPage({
    Key? key,
    required this.timeframes,
  }) : super(key: key);

  @override
  _SignalSelectorPageState createState() => _SignalSelectorPageState();
}

class _SignalSelectorPageState extends State<SignalSelectorPage> {
  List<String> currencyPairs = [];
  String selectedPair = '';
  String selectedTimeframe = '15min';
  String result = '';
  bool loading = false;

  @override
  void initState() {
    super.initState();
    fetchCurrencyPairs();
  }

  Future<void> fetchCurrencyPairs() async {
  final uri = Uri.parse('http://127.0.0.1:8000/api/pairs');

  try {
    final response = await http.get(uri);

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final List<String> pairs = List<String>.from(data['pairs']);

      setState(() {
        currencyPairs = pairs;
        selectedPair = pairs.isNotEmpty ? pairs[0] : '';
      });
    } else {
      print('Failed to load currency pairs: ${response.body}');
    }
  } catch (e) {
    print('Error loading pairs: $e');
  }
}


Future<void> fetchSignal() async {
  setState(() {
    loading = true;
    result = '';
  });

  final uri = Uri.parse(
      'http://127.0.0.1:8000/api/signal?pair=$selectedPair&tf=$selectedTimeframe');

  try {
    final response = await http.get(uri);
    print('Raw response: ${response.body}');

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final prediction = data['prediction'] ?? {};
      final indicators = prediction['indicators'] ?? {};
      final macdValue = indicators['macd']?['value']?.toStringAsFixed(4) ?? 'N/A';
      final macdSignal = indicators['macd']?['signal']?.toStringAsFixed(4) ?? 'N/A';

      setState(() {
        result = '''
Signal: ${prediction['signal'] ?? 'N/A'}
Confidence: ${prediction['confidence']?.toStringAsFixed(2) ?? 'N/A'}%
Close: ${indicators['close']?.toStringAsFixed(4) ?? 'N/A'}
RSI: ${indicators['rsi']?.toStringAsFixed(2) ?? 'N/A'}
'MACD: $macdValue (Signal: $macdSignal)'
Bollinger Bands: Upper ${indicators['bollinger_upper']?.toStringAsFixed(4) ?? 'N/A'}, Lower ${indicators['bollinger_lower']?.toStringAsFixed(4) ?? 'N/A'}
Stochastic: ${indicators['stochastic_k']?.toStringAsFixed(2) ?? 'N/A'} / ${indicators['stochastic_d']?.toStringAsFixed(2) ?? 'N/A'}
EMA20: ${indicators['ema20']?.toStringAsFixed(4) ?? 'N/A'} | EMA50: ${indicators['ema50']?.toStringAsFixed(4) ?? 'N/A'}
ADX: ${indicators['adx']?.toStringAsFixed(2) ?? 'N/A'}
CCI: ${indicators['cci']?.toStringAsFixed(2) ?? 'N/A'}
ATR: ${indicators['atr']?.toStringAsFixed(4) ?? 'N/A'}

Reasoning:
- ${(prediction['reason'] as List?)?.join('\n- ') ?? 'No reasoning provided'}
''';
      });
    } else {
      setState(() {
        result = 'Error: ${response.statusCode} - ${response.body}';
      });
    }
  } catch (e) {
    setState(() {
      result = 'Failed to connect: ${e.toString()}';
    });
  } finally {
    setState(() {
      loading = false;
    });
  }
}
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Forex Signal Selector')),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            // Currency Pair Dropdown
            currencyPairs.isEmpty
                ? CircularProgressIndicator()
                : DropdownButtonFormField<String>(
                    value: selectedPair,
                    decoration: InputDecoration(
                      labelText: 'Currency Pair',
                      border: OutlineInputBorder(),
                    ),
                    items: currencyPairs.map((pair) {
                      return DropdownMenuItem<String>(
                        value: pair,
                        child: Text(pair),
                      );
                    }).toList(),
                    onChanged: (value) {
                      setState(() {
                        selectedPair = value!;
                      });
                    },
                  ),

            const SizedBox(height: 20),

            // Timeframe Dropdown
            DropdownButtonFormField<String>(
              value: selectedTimeframe,
              decoration: InputDecoration(
                labelText: 'Timeframe',
                border: OutlineInputBorder(),
              ),
              items: widget.timeframes.map((tf) {
                return DropdownMenuItem<String>(
                  value: tf,
                  child: Text(tf),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedTimeframe = value!;
                });
              },
            ),
            const SizedBox(height: 30),

            ElevatedButton(
              onPressed: loading ? null : fetchSignal,
              child: loading
                  ? CircularProgressIndicator(
                      color: Colors.white,
                    )
                  : Text('Get Signal'),
            ),
            const SizedBox(height: 20),

            if (result.isNotEmpty)
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade400),
                  borderRadius: BorderRadius.circular(8),
                  color: Colors.grey.shade100,
                ),
                child: Text(
                  result,
                  style: TextStyle(fontSize: 14),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
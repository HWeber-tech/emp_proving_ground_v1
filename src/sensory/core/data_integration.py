    async def _get_simulated_ecb_rates(self, days_back: int) -> List[EconomicData]:
        """Generate simulated ECB rates for demonstration purposes"""
        
        data_points = []
        base_rate = 4.50  # Current ECB deposit facility rate
        
        for i in range(min(days_back // 30, 12)):  # Monthly data points
            date = datetime.now() - timedelta(days=i * 30)
            
            # Simulate rate changes with some randomness
            rate_change = np.random.normal(0, 0.1)  # Small random changes
            simulated_rate = max(0, base_rate + rate_change)
            
            data_points.append(EconomicData(
                indicator='ECB_DEPOSIT_RATE',
                value=simulated_rate,
                timestamp=date,
                frequency='monthly',
                surprise_factor=0.0,
                importance=0.9  # High importance for central bank rates
            ))
        
        return sorted(data_points, key=lambda x: x.timestamp)


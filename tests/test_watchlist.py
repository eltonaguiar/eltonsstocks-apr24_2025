import unittest
from fastapi.testclient import TestClient
from api.main import app
from api.core.database import get_db
from api.models.user import User
from api.models.watchlist import Watchlist
from sqlalchemy.orm import Session
from api.core.security import create_access_token

class TestWatchlist(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.db: Session = next(get_db())
        
        # Create a test user
        self.test_user = User(email="test@example.com", hashed_password="testpassword")
        self.db.add(self.test_user)
        self.db.commit()
        self.db.refresh(self.test_user)
        
        # Create an access token for the test user
        self.access_token = create_access_token(data={"sub": self.test_user.email})
        
    def tearDown(self):
        # Clean up the database after each test
        self.db.query(Watchlist).delete()
        self.db.query(User).delete()
        self.db.commit()
        
    def test_create_watchlist(self):
        response = self.client.post(
            "/api/watchlists/",
            json={"name": "Test Watchlist"},
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Test Watchlist")
        self.assertEqual(data["user_id"], self.test_user.id)
        
    def test_get_watchlists(self):
        # Create a watchlist for the test user
        watchlist = Watchlist(name="Test Watchlist", user_id=self.test_user.id)
        self.db.add(watchlist)
        self.db.commit()
        
        response = self.client.get(
            "/api/watchlists/",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "Test Watchlist")
        
    def test_add_stock_to_watchlist(self):
        # Create a watchlist for the test user
        watchlist = Watchlist(name="Test Watchlist", user_id=self.test_user.id)
        self.db.add(watchlist)
        self.db.commit()
        self.db.refresh(watchlist)
        
        response = self.client.post(
            f"/api/watchlists/{watchlist.id}/stocks/AAPL",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("AAPL", data["stocks"])
        
    def test_remove_stock_from_watchlist(self):
        # Create a watchlist with a stock for the test user
        watchlist = Watchlist(name="Test Watchlist", user_id=self.test_user.id, stocks=["AAPL"])
        self.db.add(watchlist)
        self.db.commit()
        self.db.refresh(watchlist)
        
        response = self.client.delete(
            f"/api/watchlists/{watchlist.id}/stocks/AAPL",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("AAPL", data["stocks"])

if __name__ == "__main__":
    unittest.main()
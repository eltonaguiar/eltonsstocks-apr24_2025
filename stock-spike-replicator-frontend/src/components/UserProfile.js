import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import handleApiError from '../utils/errorHandler';

const UserProfile = () => {
  const navigate = useNavigate();
  const { user, updateUser } = useAuth();
  const [profile, setProfile] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    const fetchProfile = async () => {
      try {
        const response = await api.get('/user/profile');
        setProfile(response.data);
      } catch (err) {
        setError(handleApiError(err));
      }
    };

    fetchProfile();
  }, [user, navigate]);

  const handleEdit = () => {
    setIsEditing(true);
    setError(null);
    setSuccessMessage(null);
  };

  const validateForm = () => {
    const errors = {};
    if (!profile.name) errors.name = 'Name is required';
    if (!profile.email) errors.email = 'Email is required';
    if (profile.email && !/\S+@\S+\.\S+/.test(profile.email)) errors.email = 'Invalid email format';
    return errors;
  };

  const handleSave = async () => {
    const formErrors = validateForm();
    if (Object.keys(formErrors).length > 0) {
      setError(formErrors);
      return;
    }

    try {
      const response = await api.put('/user/profile', profile);
      setProfile(response.data);
      updateUser(response.data);
      setIsEditing(false);
      setError(null);
      setSuccessMessage('Profile updated successfully');
    } catch (err) {
      setError(handleApiError(err));
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfile(prevProfile => ({
      ...prevProfile,
      [name]: value
    }));
  };

  if (!profile) {
    return <div className="text-center mt-5">Loading profile...</div>;
  }

  return (
    <div className="user-profile container mt-5">
      <h2 className="mb-4">User Profile</h2>
      {error && typeof error === 'string' && <div className="alert alert-danger">{error}</div>}
      {successMessage && <div className="alert alert-success">{successMessage}</div>}
      <form onSubmit={(e) => e.preventDefault()}>
        <div className="mb-3">
          <label htmlFor="name" className="form-label">Name:</label>
          <input
            type="text"
            className={`form-control ${error?.name ? 'is-invalid' : ''}`}
            id="name"
            name="name"
            value={profile.name || ''}
            onChange={handleChange}
            disabled={!isEditing}
          />
          {error?.name && <div className="invalid-feedback">{error.name}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="email" className="form-label">Email:</label>
          <input
            type="email"
            className={`form-control ${error?.email ? 'is-invalid' : ''}`}
            id="email"
            name="email"
            value={profile.email || ''}
            onChange={handleChange}
            disabled={!isEditing}
          />
          {error?.email && <div className="invalid-feedback">{error.email}</div>}
        </div>
        <div className="mb-3">
          <label htmlFor="joinDate" className="form-label">Join Date:</label>
          <input
            type="date"
            className="form-control"
            id="joinDate"
            name="joinDate"
            value={profile.joinDate || ''}
            disabled
          />
        </div>
        {isEditing ? (
          <button onClick={handleSave} className="btn btn-primary">Save</button>
        ) : (
          <button onClick={handleEdit} className="btn btn-secondary">Edit</button>
        )}
      </form>
    </div>
  );
};

export default UserProfile;